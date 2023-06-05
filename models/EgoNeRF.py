from models.coordinates import YinYangSphericalCoords
from models.tensorBase import *
from models.envmap import EnvironmentMap
import warnings
from torch.nn import AvgPool1d, AvgPool2d
from dataLoader.ray_utils import sample_pdf
from extra.test_exp_r import index2r
from math import exp, log


class YinYangAlphaGridMask(torch.nn.Module):
    def __init__(self, device, alpha_volume_yin, alpha_volume_yang):
        super(YinYangAlphaGridMask, self).__init__()
        self.device = device

        self.alpha_volume_yin = alpha_volume_yin.view(1, 1, *alpha_volume_yin.shape[-3:])
        self.alpha_volume_yang = alpha_volume_yang.view(1, 1, *alpha_volume_yang.shape[-3:])

    def sample_alpha(self, norm_samples):
        alpha_vals = torch.empty_like(norm_samples[:, 0])
        is_yin_grid = norm_samples[:, -1] == 0
        alpha_vals[is_yin_grid] = F.grid_sample(self.alpha_volume_yin, norm_samples[is_yin_grid][:, :3].view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        alpha_vals[~is_yin_grid] = F.grid_sample(self.alpha_volume_yang, norm_samples[~is_yin_grid][:, 3:6].view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals


class EgoNeRF(TensorBase):
    def __init__(self, aabb, gridSize, device, coordinates, **kargs):
        super(EgoNeRF, self).__init__(aabb, gridSize, device, coordinates, **kargs)
        self.matMode_yin = [[0, 1], [0, 2], [1, 2]]
        self.vecMode_yin = [2, 1, 0]
        self.matMode_yang = [[0, 1], [0, 2], [1, 2]]
        self.vecMode_yang = [2, 1, 0]

        self.density_plane_yin = None
        self.density_line_yin = None
        self.app_plane_yin = None
        self.app_line_yin = None
        self.basis_mat_yin = None

        self.density_plane_yang = None
        self.density_line_yang = None
        self.app_plane_yang = None
        self.app_line_yang = None
        self.basis_mat_yang = None

        self.coarse_sigma_line_yin = [None, None, None]
        self.coarse_sigma_plane_yin = [None, None, None]
        self.coarse_sigma_line_yang = [None, None, None]
        self.coarse_sigma_plane_yang = [None, None, None]

        self.init_svd_volume(gridSize[0], device)
        if self.coarse_sigma_grid_update_rule is not None:
            self.init_coarse_density_volume(self.coarse_sigma_grid_update_rule)

    def sample_ray_exp(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples
        near, far = self.near_far
        if not self.coordinates.interval_th:
            ratio = 1 + (pi / 2.) / N_samples  # approximate ratio
            # r0 = max((far - near) * (ratio - 1) / (pow(ratio, N_samples) - 1), 0.002)
            r0 = (far - near) * (ratio - 1) / (pow(ratio, N_samples) - 1)
            rng = torch.arange(N_samples)[None].float()
            if is_train:
                rng = rng.repeat(rays_d.shape[-2], 1)
                rng += torch.rand_like(rng)
            interpx = (near + torch.pow(ratio, rng) @ torch.tril(torch.ones(N_samples, N_samples), diagonal=-1).T * r0).to(rays_o.device)
        else:
            rng = torch.arange(N_samples).float()
            ratio = exp(log((far - near) / self.coordinates.r0) / (N_samples - 1))
            r = index2r(self.coordinates.r0, ratio, rng)
            interval = r[1:] - r[:-1]
            interval_cum = torch.cumsum(interval, dim=0)
            interval_less_than_r0 = interval <= self.coordinates.r0
            r[:interval_less_than_r0.sum() + 1] = torch.arange(interval_less_than_r0.sum() + 1) * self.coordinates.r0
            r[interval_less_than_r0.sum() + 1:] = r[interval_less_than_r0.sum() + 1:] + self.coordinates.r0 * interval_less_than_r0.sum() - interval_cum[interval_less_than_r0.sum() - 1]
            r = r.repeat(rays_d.shape[-2], 1)
            if is_train:
                interval = r[:, 1:] - r[:, :-1]
                interval = torch.cat([interval, interval[:, -1:]], dim=-1)
                r = r + interval * torch.rand_like(r)
            interpx = (near + r).to(rays_o.device)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def init_coarse_density_volume(self, device):
        if self.coarse_sigma_grid_update_rule == 'conv':
            self.update_coarse_sigma_grid()
        else:
            self.coarse_sigma_plane_yin, self.coarse_sigma_line_yin, self.coarse_sigma_plane_yang, self.coarse_sigma_line_yang = self.init_one_svd(self.density_n_compo, self.coarse_gridSize, 0.1, device)
            raise NotImplementedError

    def init_svd_volume(self, res, device):
        self.density_plane_yin, self.density_line_yin, self.density_plane_yang, self.density_line_yang = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane_yin, self.app_line_yin, self.app_plane_yang, self.app_line_yang = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat_yin = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        self.basis_mat_yang = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef_yin, line_coef_yin = [], []
        plane_coef_yang, line_coef_yang = [], []

        assert len(self.vecMode_yin) == len(self.matMode_yin)

        for i in range(len(self.vecMode_yin)):
            vec_id_yin, vec_id_yang = self.vecMode_yin[i], self.vecMode_yang[i]
            mat_id_yin_0, mat_id_yin_1 = self.matMode_yin[i]
            mat_id_yang_0, mat_id_yang_1 = self.matMode_yang[i]
            plane_coef_yin.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_yin_1], gridSize[mat_id_yin_0]))))
            line_coef_yin.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id_yin], 1))))
            plane_coef_yang.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_yang_1], gridSize[mat_id_yang_0]))))
            line_coef_yang.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id_yang], 1))))

        return torch.nn.ParameterList(plane_coef_yin).to(device), torch.nn.ParameterList(line_coef_yin).to(device),\
               torch.nn.ParameterList(plane_coef_yang).to(device), torch.nn.ParameterList(line_coef_yang).to(device)

    def update_coarse_sigma_grid(self):
        if self.coarse_sigma_grid_update_rule == 'conv':
            assert len(self.density_plane_yin) == len(self.density_plane_yang)
            for i in range(len(self.density_plane_yin)):
                self.coarse_sigma_plane_yin[i] = AvgPool2d(kernel_size=2, stride=2)(self.density_plane_yin[i])
                self.coarse_sigma_line_yin[i] = AvgPool1d(kernel_size=2, stride=2)(self.density_line_yin[i].squeeze(dim=-1)).unsqueeze(dim=-1)
                self.coarse_sigma_plane_yang[i] = AvgPool2d(kernel_size=2, stride=2)(self.density_plane_yang[i])
                self.coarse_sigma_line_yang[i] = AvgPool1d(kernel_size=2, stride=2)(self.density_line_yang[i].squeeze(dim=-1)).unsqueeze(dim=-1)
        else:
            raise NotImplementedError

    def init_envmap(self, envmap_res_H, init_strategy='zero', device='cuda'):
        self.envmap = EnvironmentMap(h=envmap_res_H, init_strategy=init_strategy, device=device)
        # self.envmap = torch.nn.Parameter(torch.rand((envmap_res_H, 2 * envmap_res_H))).to(device)  # equirectangular

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001, lr_init_envmap=0.1):
        grad_vars = [
            {'params': self.density_line_yin, 'lr': lr_init_spatialxyz},
            {'params': self.density_plane_yin, 'lr': lr_init_spatialxyz},
            {'params': self.app_line_yin, 'lr': lr_init_spatialxyz},
            {'params': self.app_plane_yin, 'lr': lr_init_spatialxyz},
            {'params': self.basis_mat_yin.parameters(), 'lr': lr_init_network},
            {'params': self.density_line_yang, 'lr': lr_init_spatialxyz},
            {'params': self.density_plane_yang, 'lr': lr_init_spatialxyz},
            {'params': self.app_line_yang, 'lr': lr_init_spatialxyz},
            {'params': self.app_plane_yang, 'lr': lr_init_spatialxyz},
            {'params': self.basis_mat_yang.parameters(), 'lr': lr_init_network}
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        if self.envmap is not None:
            grad_vars += [{'params': self.envmap.emission, 'lr': lr_init_envmap}]  # TODO: lr for envmap
        return grad_vars

    def save(self, path, global_step):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict(), 'global_step': global_step}
        if self.alphaMask is not None:
            alpha_volume_yin = self.alphaMask.alpha_volume_yin.bool().cpu().numpy()
            ckpt.update({'alphaMask_yin.shape': alpha_volume_yin.shape})
            ckpt.update({'alphaMask_yin.mask': np.packbits(alpha_volume_yin.reshape(-1))})
            alpha_volume_yang = self.alphaMask.alpha_volume_yang.bool().cpu().numpy()
            ckpt.update({'alphaMask_yang.shape': alpha_volume_yang.shape})
            ckpt.update({'alphaMask_yang.mask': np.packbits(alpha_volume_yang.reshape(-1))})
        if self.envmap is not None:
            envmap_emission = self.envmap.emission.detach().cpu().numpy()
            ckpt.update({'envmap.emission': envmap_emission})
            ckpt.update({'envmap_res_H': self.envmap.emission.shape[2]})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask_yin.shape' in ckpt.keys():
            length = np.prod(ckpt['alphaMask_yin.shape'])
            alpha_volume_yin = torch.from_numpy(np.unpackbits(ckpt['alphaMask_yin.mask'])[:length].reshape(ckpt['alphaMask_yin.shape']))
            length = np.prod(ckpt['alphaMask_yang.shape'])
            alpha_volume_yang = torch.from_numpy(np.unpackbits(ckpt['alphaMask_yang.mask'])[:length].reshape(ckpt['alphaMask_yang.shape']))
            self.alphaMask = YinYangAlphaGridMask(self.device, alpha_volume_yin.float().to(self.device), alpha_volume_yang.float().to(self.device))
        if self.envmap is not None:
            self.envmap = EnvironmentMap(h=ckpt['envmap_res_H'])
            self.envmap.load_envmap(emission=ckpt['envmap.emission'], device=self.device)
        self.load_state_dict(ckpt['state_dict'])
        if self.coarse_sigma_grid_update_rule == 'conv':
            self.update_coarse_sigma_grid()
        return ckpt['global_step']

    def vectorDiffs(self, vector_comps):
        total = 0
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line_yin) + self.vectorDiffs(self.app_line_yin) + self.vectorDiffs(self.density_line_yang) + self.vectorDiffs(self.app_line_yang)
    def coarse_sigma_L1(self):
        total = 0
        assert len(self.coarse_sigma_plane_yin) == len(self.coarse_sigma_plane_yang)
        for idx in range(len(self.coarse_sigma_plane_yin)):
            total = total + torch.mean(torch.abs(self.coarse_sigma_plane_yin[idx])) + torch.mean(torch.abs(self.coarse_sigma_line_yin[idx]))
            total = total + torch.mean(torch.abs(self.coarse_sigma_plane_yang[idx])) + torch.mean(torch.abs(self.coarse_sigma_line_yang[idx]))
        return total

    def density_L1(self):
        total = 0
        assert len(self.density_plane_yin) == len(self.density_plane_yang)
        for idx in range(len(self.density_plane_yin)):
            total = total + torch.mean(torch.abs(self.density_plane_yin[idx])) + torch.mean(torch.abs(self.density_line_yin[idx]))
            total = total + torch.mean(torch.abs(self.density_plane_yang[idx])) + torch.mean(torch.abs(self.density_line_yang[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        assert len(self.density_plane_yin) == len(self.density_plane_yang)
        for idx in range(len(self.density_plane_yin)):
            total = total + reg(self.density_plane_yin[idx]) * 1e-2  # + reg(self.density_line_yin[idx]) * 1e-3
            total = total + reg(self.density_plane_yang[idx]) * 1e-2  # + reg(self.density_line_yang[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        assert len(self.app_plane_yin) == len(self.app_plane_yang)
        for idx in range(len(self.app_plane_yin)):
            total = total + reg(self.app_plane_yin[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
            total = total + reg(self.app_plane_yang[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
        return total

    def compute_coarse_densityfeature(self, coords_sampled, coarse_sigma_grid_update_rule='conv'):
        # TODO: rename (compute -> get?)
        yin_filter = coords_sampled[..., -1] == 0  # the last coordinate contains if it is yin or yang
        yang_filter = torch.logical_not(yin_filter)
        is_yin_empty = torch.sum(yin_filter) == 0
        is_yang_empty = torch.sum(yang_filter) == 0
        coords_yin = coords_sampled[yin_filter][:, :3]
        coords_yang = coords_sampled[yang_filter][:, 3:6]

        # Yin Grid
        if not is_yin_empty:
            coordinate_plane_yin = torch.stack((
                coords_yin[..., self.matMode_yin[0]],
                coords_yin[..., self.matMode_yin[1]],
                coords_yin[..., self.matMode_yin[2]]
            )).detach().view(3, -1, 1, 2)
            coordinate_line_yin = torch.stack((
                coords_yin[..., self.vecMode_yin[0]],
                coords_yin[..., self.vecMode_yin[1]],
                coords_yin[..., self.vecMode_yin[2]])
            )
            coordinate_line_yin = torch.stack((
                torch.zeros_like(coordinate_line_yin), coordinate_line_yin), dim=-1
            ).detach().view(3, -1, 1, 2)

        # Yang Grid
        if not is_yang_empty:
            coordinate_plane_yang = torch.stack((
                coords_yang[..., self.matMode_yang[0]],
                coords_yang[..., self.matMode_yang[1]],
                coords_yang[..., self.matMode_yang[2]]
            )).detach().view(3, -1, 1, 2)
            coordinate_line_yang = torch.stack((
                coords_yang[..., self.vecMode_yang[0]],
                coords_yang[..., self.vecMode_yang[1]],
                coords_yang[..., self.vecMode_yang[2]])
            )
            coordinate_line_yang = torch.stack((
                torch.zeros_like(coordinate_line_yang), coordinate_line_yang), dim=-1
            ).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((coords_sampled.shape[:-1]), device=coords_sampled.device)

        assert len(self.coarse_sigma_plane_yin) == len(self.coarse_sigma_plane_yang)
        for idx_plane in range(len(self.density_plane_yin)):
            if not is_yin_empty:
                plane_coef_point_yin = F.grid_sample(self.coarse_sigma_plane_yin[idx_plane], coordinate_plane_yin[[idx_plane]],
                                                     align_corners=True).view(-1, *coords_yin.shape[:1])
                line_coef_point_yin = F.grid_sample(self.coarse_sigma_line_yin[idx_plane], coordinate_line_yin[[idx_plane]],
                                                    align_corners=True).view(-1, *coords_yin.shape[:1])
                sigma_feature[yin_filter] = sigma_feature[yin_filter] + F.relu(torch.sum(plane_coef_point_yin * line_coef_point_yin, dim=0))
            if not is_yang_empty:
                plane_coef_point_yang = F.grid_sample(self.coarse_sigma_plane_yang[idx_plane], coordinate_plane_yang[[idx_plane]],
                                                      align_corners=True).view(-1, *coords_yang.shape[:1])
                line_coef_point_yang = F.grid_sample(self.coarse_sigma_line_yang[idx_plane], coordinate_line_yang[[idx_plane]],
                                                     align_corners=True).view(-1, *coords_yang.shape[:1])
                sigma_feature[yang_filter] = sigma_feature[yang_filter] + F.relu(torch.sum(plane_coef_point_yang * line_coef_point_yang, dim=0))
        return sigma_feature

    def compute_densityfeature(self, coords_sampled):
        yin_filter = coords_sampled[..., -1] == 0  # the last coordinate contains if it is yin or yang
        yang_filter = torch.logical_not(yin_filter)
        is_yin_empty = torch.sum(yin_filter) == 0
        is_yang_empty = torch.sum(yang_filter) == 0
        coords_yin = coords_sampled[yin_filter][:, :3]
        coords_yang = coords_sampled[yang_filter][:, 3:6]

        # Yin Grid
        if not is_yin_empty:
            coordinate_plane_yin = torch.stack((
                coords_yin[..., self.matMode_yin[0]],
                coords_yin[..., self.matMode_yin[1]],
                coords_yin[..., self.matMode_yin[2]]
            )).detach().view(3, -1, 1, 2)
            coordinate_line_yin = torch.stack((
                coords_yin[..., self.vecMode_yin[0]],
                coords_yin[..., self.vecMode_yin[1]],
                coords_yin[..., self.vecMode_yin[2]])
            )
            coordinate_line_yin = torch.stack((
                torch.zeros_like(coordinate_line_yin), coordinate_line_yin), dim=-1
            ).detach().view(3, -1, 1, 2)

        # Yang Grid
        if not is_yang_empty:
            coordinate_plane_yang = torch.stack((
                coords_yang[..., self.matMode_yang[0]],
                coords_yang[..., self.matMode_yang[1]],
                coords_yang[..., self.matMode_yang[2]]
            )).detach().view(3, -1, 1, 2)
            coordinate_line_yang = torch.stack((
                coords_yang[..., self.vecMode_yang[0]],
                coords_yang[..., self.vecMode_yang[1]],
                coords_yang[..., self.vecMode_yang[2]])
            )
            coordinate_line_yang = torch.stack((
                torch.zeros_like(coordinate_line_yang), coordinate_line_yang), dim=-1
            ).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((coords_sampled.shape[:-1]), device=coords_sampled.device)

        assert len(self.density_plane_yin) == len(self.density_plane_yang)
        for idx_plane in range(len(self.density_plane_yin)):
            if not is_yin_empty:
                plane_coef_point_yin = F.grid_sample(self.density_plane_yin[idx_plane], coordinate_plane_yin[[idx_plane]],
                                                     align_corners=True).view(-1, *coords_yin.shape[:1])
                line_coef_point_yin = F.grid_sample(self.density_line_yin[idx_plane], coordinate_line_yin[[idx_plane]],
                                                    align_corners=True).view(-1, *coords_yin.shape[:1])
                sigma_feature[yin_filter] = sigma_feature[yin_filter] + F.relu(torch.sum(plane_coef_point_yin * line_coef_point_yin, dim=0))
            if not is_yang_empty:
                plane_coef_point_yang = F.grid_sample(self.density_plane_yang[idx_plane], coordinate_plane_yang[[idx_plane]],
                                                      align_corners=True).view(-1, *coords_yang.shape[:1])
                line_coef_point_yang = F.grid_sample(self.density_line_yang[idx_plane], coordinate_line_yang[[idx_plane]],
                                                     align_corners=True).view(-1, *coords_yang.shape[:1])
                sigma_feature[yang_filter] = sigma_feature[yang_filter] + F.relu(torch.sum(plane_coef_point_yang * line_coef_point_yang, dim=0))
        return sigma_feature

    def compute_appfeature(self, coords_sampled):
        yin_filter = coords_sampled[..., -1] == 0  # the last coordinate contains if it is yin or yang
        yang_filter = torch.logical_not(yin_filter)
        is_yin_empty = torch.sum(yin_filter) == 0
        is_yang_empty = torch.sum(yang_filter) == 0
        coords_yin = coords_sampled[yin_filter][:, :3]
        coords_yang = coords_sampled[yang_filter][:, 3:6]

        if not is_yin_empty:
            coordinate_plane_yin = torch.stack((
                coords_yin[..., self.matMode_yin[0]],
                coords_yin[..., self.matMode_yin[1]],
                coords_yin[..., self.matMode_yin[2]])
            ).detach().view(3, -1, 1, 2)
            coordinate_line_yin = torch.stack((
                coords_yin[..., self.vecMode_yin[0]],
                coords_yin[..., self.vecMode_yin[1]],
                coords_yin[..., self.vecMode_yin[2]])
            )
            coordinate_line_yin = torch.stack((
                torch.zeros_like(coordinate_line_yin), coordinate_line_yin), dim=-1
            ).detach().view(3, -1, 1, 2)

        if not is_yang_empty:
            coordinate_plane_yang = torch.stack((
                coords_yang[..., self.matMode_yang[0]],
                coords_yang[..., self.matMode_yang[1]],
                coords_yang[..., self.matMode_yang[2]])
            ).detach().view(3, -1, 1, 2)
            coordinate_line_yang = torch.stack((
                coords_yang[..., self.vecMode_yang[0]],
                coords_yang[..., self.vecMode_yang[1]],
                coords_yang[..., self.vecMode_yang[2]])
            )
            coordinate_line_yang = torch.stack((
                torch.zeros_like(coordinate_line_yang), coordinate_line_yang), dim=-1
            ).detach().view(3, -1, 1, 2)

        app_feature = torch.zeros((*coords_sampled.shape[:-1], self.app_dim), device=coords_sampled.device)

        plane_coef_point_yin, line_coef_point_yin, plane_coef_point_yang, line_coef_point_yang = [], [], [], []

        assert len(self.app_plane_yin) == len(self.app_plane_yang)
        for idx_plane in range(len(self.app_plane_yin)):
            if not is_yin_empty:
                plane_coef_point_yin.append(F.grid_sample(
                    self.app_plane_yin[idx_plane], coordinate_plane_yin[[idx_plane]], align_corners=True
                ).view(-1, *coords_yin.shape[:1]))
                line_coef_point_yin.append(F.grid_sample(
                    self.app_line_yin[idx_plane], coordinate_line_yin[[idx_plane]], align_corners=True
                ).view(-1, *coords_yin.shape[:1]))
            if not is_yang_empty:
                plane_coef_point_yang.append(F.grid_sample(
                    self.app_plane_yang[idx_plane], coordinate_plane_yang[[idx_plane]], align_corners=True
                ).view(-1, *coords_yang.shape[:1]))
                line_coef_point_yang.append(F.grid_sample(
                    self.app_line_yang[idx_plane], coordinate_line_yang[[idx_plane]], align_corners=True
                ).view(-1, *coords_yang.shape[:1]))
        if not is_yin_empty:
            plane_coef_point_yin, line_coef_point_yin = torch.cat(plane_coef_point_yin), torch.cat(line_coef_point_yin)
            app_feature[yin_filter] = self.basis_mat_yin((plane_coef_point_yin * line_coef_point_yin).T)
        if not is_yang_empty:
            plane_coef_point_yang, line_coef_point_yang = torch.cat(plane_coef_point_yang), torch.cat(line_coef_point_yang)
            app_feature[yang_filter] = self.basis_mat_yang((plane_coef_point_yang * line_coef_point_yang).T)
        return app_feature

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        from models.coordinates import YinYangSphericalCoords
        assert isinstance(self.coordinates, YinYangSphericalCoords)
        assert len(self.vecMode_yin) == len(self.vecMode_yang)
        for i in range(len(self.vecMode_yin)):
            vec_id = self.vecMode_yin[i]
            mat_id_0, mat_id_1 = self.matMode_yin[i]
            plane_coef[i] = self.coordinates.up_sampling_VM(plane_coef[i].data, res_target=res_target, ids=[mat_id_1, mat_id_0])
            line_coef[i] = self.coordinates.up_sampling_VM(line_coef[i].data, res_target=res_target, ids=[vec_id])
        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane_yin, self.app_line_yin = self.up_sampling_VM(self.app_plane_yin, self.app_line_yin, res_target)
        self.density_plane_yin, self.density_line_yin = self.up_sampling_VM(self.density_plane_yin, self.density_line_yin, res_target)

        self.app_plane_yang, self.app_line_yang = self.up_sampling_VM(self.app_plane_yang, self.app_line_yang, res_target)
        self.density_plane_yang, self.density_line_yang = self.up_sampling_VM(self.density_plane_yang, self.density_line_yang, res_target)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        warnings.warn("\n------------getDenseAlpha is deprecated----------\n", DeprecationWarning)
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        norm_coords_locs = samples * 2 - 1

        yin_coords_locs = torch.cat((
            norm_coords_locs,
            torch.zeros_like(norm_coords_locs),
            torch.zeros_like(norm_coords_locs[..., -1:]),
            ), -1).to(self.device)
        yang_coords_locs = torch.cat((
            torch.zeros_like(norm_coords_locs),
            norm_coords_locs,
            torch.ones_like(norm_coords_locs[..., -1:]),
            ), -1).to(self.device)

        alpha_yin = torch.zeros_like(norm_coords_locs[..., 0])
        alpha_yang = torch.zeros_like(norm_coords_locs[..., 0])
        for i in range(gridSize[0]):
            alpha_yin[i] = self.compute_alpha(rearrange(yin_coords_locs[i], 'h w c -> (h w) c'), self.stepSize).view((gridSize[1], gridSize[2]))
            alpha_yang[i] = self.compute_alpha(rearrange(yang_coords_locs[i], 'h w c -> (h w) c'), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha_yin, alpha_yang

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=None):
        warnings.warn("\n------------updateAlphaMask is deprecated----------\n", DeprecationWarning)
        alpha_yin, alpha_yang = self.getDenseAlpha(gridSize)

        alpha_yin = alpha_yin.clamp(0,1).transpose(0,2).contiguous()[None,None]
        alpha_yang = alpha_yang.clamp(0,1).transpose(0,2).contiguous()[None,None]

        total_voxels = gridSize[0] * gridSize[1] * gridSize[2] * 2

        ks = 3
        alpha_yin = F.max_pool3d(alpha_yin, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha_yin[alpha_yin>=self.alphaMask_thres] = 1
        alpha_yin[alpha_yin<self.alphaMask_thres] = 0

        alpha_yang = F.max_pool3d(alpha_yang, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha_yang[alpha_yang>=self.alphaMask_thres] = 1
        alpha_yang[alpha_yang<self.alphaMask_thres] = 0

        self.alphaMask = YinYangAlphaGridMask(self.device, alpha_yin, alpha_yang)

        total = torch.sum(alpha_yin) + torch.sum(alpha_yang)
        print(f"alpha rest %%%f"%(total/total_voxels*100))

    def forward(
            self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, n_coarse=-1, n_fine=0,
            exp_sampling=False, pretrain_envmap=False, pivotal_sample_th=0., resampling=False, use_coarse_sample=True,
            interval_th=False
    ):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if pretrain_envmap:
            env_map = self.envmap.get_radiance(viewdirs)
            return env_map

        # 1) sample coarse ray points (N_rays, n_coarse, 3)
        if ndc_ray:
            raise NotImplementedError
        else:
            if exp_sampling:
                coarse_xyz_sampled, coarse_z_vals, _ = self.sample_ray_exp(
                    rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=n_coarse
                )
            else:
                coarse_xyz_sampled, coarse_z_vals, _ = self.sample_ray(
                    rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=n_coarse
                )
            # dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            if not is_train:
                coarse_z_vals = coarse_z_vals[0].repeat(coarse_xyz_sampled.shape[0], 1)
            coarse_dists = coarse_z_vals[..., 1:] - coarse_z_vals[..., :-1]
            coarse_dists = torch.cat((coarse_dists, coarse_dists[..., -1:]), dim=-1)  # (N_rays, N_samples)

        coarse_coords_sampled = self.coordinates.from_cartesian(coarse_xyz_sampled)
        # coords_sampled = self.coordinates.normalize_coord(coords_sampled)
        assert isinstance(self.coordinates, YinYangSphericalCoords), "normalize_coord with downsampling is supported only for YinYangSphericalCoords currently."
        coarse_coords_sampled = self.coordinates.normalize_coord(coarse_coords_sampled, downsample=2)

        if resampling:
            # 2) Compute weights from coarse samples
            coarse_sigma_feature = self.compute_coarse_densityfeature(coarse_coords_sampled)
            coarse_sigma = self.feature2density(coarse_sigma_feature)
            coarse_alpha, coarse_weight, coarse_bg_weight = raw2alpha(coarse_sigma, coarse_dists * self.distance_scale)  # TODO: need distance_scale at here?

            # 3) Resample fine samples from weight pdf of coarse samples
            coarse_z_vals_mid = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
            fine_z_samples = sample_pdf(coarse_z_vals_mid, coarse_weight[..., 1:-1], N_samples=n_fine, is_train=is_train)
            fine_z_samples = fine_z_samples.detach()
            # fine_z_vals, _ = torch.sort(fine_z_samples, -1)
            if use_coarse_sample:
                fine_z_vals, _ = torch.sort(torch.cat([coarse_z_vals, fine_z_samples], -1), -1)
            else:
                fine_z_vals, _ = torch.sort(fine_z_samples, -1)

            fine_dists = fine_z_vals[..., 1:] - fine_z_vals[..., :-1]
            fine_dists = torch.cat((fine_dists, fine_dists[..., -1:]), dim=-1)

            fine_xyz_sampled = rays_chunk[:, None, :3] + viewdirs[..., None, :] * fine_z_vals[..., None]
            fine_coords_sampled = self.coordinates.from_cartesian(fine_xyz_sampled)
            fine_coords_sampled = self.coordinates.normalize_coord(fine_coords_sampled)

            viewdirs = viewdirs.view(-1, 1, 3).expand(fine_xyz_sampled.shape)

            # 4) volume render with fine samples on camera rays
            fine_sigma_feature = self.compute_densityfeature(fine_coords_sampled)
            fine_sigma = self.feature2density(fine_sigma_feature)
            fine_alpha, fine_weight, fine_bg_weight = raw2alpha(fine_sigma, fine_dists * self.distance_scale)

            app_features = self.compute_appfeature(fine_coords_sampled)
            rgb = self.renderModule(fine_coords_sampled, viewdirs, app_features)

            weight = fine_weight
            alpha = fine_alpha
            bg_weight = fine_bg_weight
            z_vals = fine_z_vals
            dists = fine_dists

        else:
            coarse_sigma_feature = self.compute_densityfeature(coarse_coords_sampled)
            coarse_sigma = self.feature2density(coarse_sigma_feature)
            coarse_alpha, coarse_weight, coarse_bg_weight = raw2alpha(coarse_sigma, coarse_dists * self.distance_scale)  # TODO: need distance_scale at here?

            viewdirs = viewdirs.view(-1, 1, 3).expand(coarse_xyz_sampled.shape)
            # app_mask = torch.ones_like(coarse_weight, dtype=torch.bool)
            app_features = self.compute_appfeature(coarse_coords_sampled)
            rgb = self.renderModule(coarse_coords_sampled, viewdirs, app_features)
            weight = coarse_weight
            alpha = coarse_alpha
            bg_weight = coarse_bg_weight
            z_vals = coarse_z_vals
            dists = coarse_dists

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
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        mid_points = (z_vals[..., 1:] + z_vals[..., :-1]) * 0.5

        return rgb_map, depth_map, bg_map, env_map, alpha  # rgb, sigma, alpha, weight, bg_weight
