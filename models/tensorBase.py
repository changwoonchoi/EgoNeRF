from math import pi
from einops import rearrange
import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from .coordinates import Coordinates
from warnings import warn
from .envmap import EnvironmentMap


def positional_encoding(positions, freqs):
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])

    def sample_alpha(self, norm_samples):
        alpha_vals = F.grid_sample(self.alpha_volume, norm_samples.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChannel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChannel + 3 + inChannel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChannel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChannel
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)
        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChannel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChannel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class TensorBase(torch.nn.Module):
    def __init__(
            self, aabb, gridSize, device, coordinates: Coordinates, density_n_comp=8, appearance_n_comp=24,
            app_dim=27, shadingMode='MLP_PE', alphaMask=None, near_far=[2.0, 6.0], density_shift=-10,
            alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001, pos_pe=6, view_pe=6,
            fea_pe=6, featureC=128, step_ratio=2.0, fea2denseAct='softplus', use_envmap=False, envmap_res_H=1000,
            envmap=None, coarse_sigma_grid_update_rule=None, coarse_sigma_grid_reso=None, interval_th=False
    ):
        super().__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = None
        self.vecMode = None
        self.comp_w = [1, 1, 1]

        # TODO: currently define tensors for only VM decomposition
        self.density_plane = None
        self.density_line = None
        self.app_plane = None
        self.app_line = None
        self.basis_mat = None

        self.envmap = None

        # self.init_svd_volume(gridSize[0], device) --> moved to children classes
        if use_envmap:
            if envmap is None:
                self.init_envmap(envmap_res_H, init_strategy='random', device=device)
            else:
                self.envmap = EnvironmentMap(h=envmap.emission.shape[2])
                self.envmap.load_envmap(envmap.emission, device=device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

        self.coordinates = coordinates

        self.coarse_sigma_grid_update_rule = coarse_sigma_grid_update_rule

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbHalfDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize))) / 2.0 # use half of diagonal since our scene is egocentric
        self.nSamples = int((self.aabbHalfDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def init_envmap(self, envmap_res_H, init_strategy, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        warn('This method is deprecated, use coordinates.normalized_coord instead.', DeprecationWarning, stacklevel=2)
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,

            'coordinates': self.coordinates,
            'use_envmap': self.envmap is not None,
            'envmap': self.envmap,
            'coarse_sigma_grid_update_rule': self.coarse_sigma_grid_update_rule,
        }

    def save(self, path, global_step):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict(), 'global_step': global_step}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            # ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        if self.envmap is not None:
            envmap_emission = self.envmap.emission.detach().cpu().numpy()
            ckpt.update({'envmap.emission': envmap_emission})
            ckpt.update({'envmap_res_H': self.envmap.emission.shape[2]})
        torch.save(ckpt, path)

    def load(self, ckpt):
        # if 'alphaMask.aabb' in ckpt.keys():
        if 'alphaMask.shape' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            # self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
            self.alphaMask = AlphaGridMask(self.device, alpha_volume.float().to(self.device))
        if self.envmap is not None:
            self.envmap = EnvironmentMap(h=ckpt['envmap_res_H'])
            self.envmap.load_envmap(emission=ckpt['envmap.emission'], device=self.device)
        self.load_state_dict(ckpt['state_dict'])
        return ckpt['global_step']

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng)
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_exp(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        near, far = self.near_far
        ratio = 1 + pi / N_samples # approximate ratio
        r0 = max((far - near) * (ratio - 1) / (pow(ratio, N_samples) - 1), 0.002)
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng)
        interpx = (near + torch.pow(ratio, rng) @ torch.tril(torch.ones(N_samples, N_samples), diagonal=-1).T * r0).to(rays_o.device)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)
        # mask_outbbox[...] = False

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        norm_coords_locs = samples * 2 - 1

        alpha = torch.zeros_like(norm_coords_locs[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(rearrange(norm_coords_locs[i], 'h w c -> (h w) c'), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        alpha = self.getDenseAlpha(gridSize)
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, alpha)

        total = torch.sum(alpha)
        print(f"alpha rest %%%f"%(total/total_voxels*100))

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, all_depths=None, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        if all_depths is None:
            return all_rays[mask_filtered], all_rgbs[mask_filtered]
        else:
            return all_rays[mask_filtered], all_rgbs[mask_filtered], all_depths[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, norm_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(norm_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(norm_locs[:,0], dtype=bool)

        sigma = torch.zeros(norm_locs.shape[:-1], device=norm_locs.device)

        if alpha_mask.any():
            sigma_feature = self.compute_densityfeature(norm_locs[alpha_mask])
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma*length).view(norm_locs.shape[:-1])
        return alpha

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, exp_sampling=False, pretrain_envmap=False):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if pretrain_envmap:
            env_map = self.envmap.get_radiance(viewdirs)
            return env_map

        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            if exp_sampling:
                xyz_sampled, z_vals, ray_valid = self.sample_ray_exp(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
            else:
                xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
            # dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat((dists, dists[..., -1:]), dim=-1)  # (N_rays, N_samples)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        coords_sampled = self.coordinates.from_cartesian(xyz_sampled)
        coords_sampled = self.coordinates.normalize_coord(coords_sampled)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(coords_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(coords_sampled.shape[:-1], device=coords_sampled.device)
        rgb = torch.zeros((*coords_sampled.shape[:2], 3), device=coords_sampled.device)

        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(coords_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(coords_sampled[app_mask])
            valid_rgbs = self.renderModule(coords_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        # if white_bg or (is_train and torch.rand((1,))<0.5):
        #     rgb_map = rgb_map + (1. - acc_map[..., None])
        bg_map = None
        env_map = None
        if self.envmap is not None:
            alpha = torch.cat((alpha, torch.ones_like(alpha[..., :1])), dim=-1)
            env_map = self.envmap.get_radiance(viewdirs[:, 0, :])
            bg_map = bg_weight * env_map
            rgb_map = rgb_map + bg_map
            # rgb_map = rgb_map + bg_weight * self.envmap.get_radiance(viewdirs[:, 0, :]).view(-1, 3)

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            # depth_map = depth_map + torch.squeeze(bg_weight) * z_vals[:, -1]
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]  # TODO: check this line

        return rgb_map, depth_map, bg_map, env_map, alpha  # rgb, sigma, alpha, weight, bg_weight
