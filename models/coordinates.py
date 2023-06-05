from math import log, pi, sqrt, exp
import torch
import torch.nn.functional as F
from extra.test_exp_r import index2r


class Coordinates:
    def __init__(self, device, aabb):
        self.device = device
        self.update_aabb(aabb)

    def normalize_coord(self, sample_points):
        pass

    def from_cartesian(self, xyz_points):
        pass

    def update_aabb(self, new_aabb):
        pass

    def get_normalized_range(self, new_aabb):
        pass

    def N_to_reso(self, n_voxels, bbox):
        pass

    def up_sampling_VM(self, weights, res_target, ids):
        """
        default: linear interpolation
        """
        assert len(ids) == 1 or len(ids) == 2, 'len(ids) should be 1 or 2!'

        if len(ids) == 1:
            target_size = (res_target[ids[0]], 1)
        elif len(ids) == 2:
            target_size = (res_target[ids[0]], res_target[ids[1]])

        return torch.nn.Parameter(
            F.interpolate(weights, size=target_size, mode='bilinear', align_corners=True))

    def set_resolution(self, resolution):
        # raise NotImplementedError('set_resolution is deprecated')
        self.resolution = resolution


class CartesianCoords(Coordinates):
    """
    Cartesian Coordinates[xyz]
    """
    def normalize_coord(self, sample_points):
        return (sample_points - self.aabb[0]) * self.invgridSize * 2 - 1

    def from_cartesian(self, xyz_points):
        return xyz_points

    def update_aabb(self, new_aabb):
        self.aabb = new_aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize

    def get_normalized_range(self, new_aabb):
        xyz_min, xyz_max = new_aabb
        return (xyz_min - self.aabb[0]) * self.invgridSize, (xyz_max - self.aabb[0]) * self.invgridSize

    def N_to_reso(self, n_voxels, bbox):
        # calculate resolution s.t. each voxel looks like a cube
        xyz_min, xyz_max = bbox
        dim = len(xyz_min)
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()


class GenericSphericalCoords(Coordinates):
    """
    Generic Spherical Coordinates including uniform r grid and exponential r grid. [r, theta, phi]
    """
    def __init__(self, device, aabb, exp_r=False, N_voxel=None, r0=None, interval_th=False):
        self.center = aabb.to(device).sum(0).div(2)
        super(GenericSphericalCoords, self).__init__(device, aabb)
        self.exp_r = exp_r
        self.interval_th = interval_th
        self.set_resolution(resolution=self.N_to_reso(N_voxel, aabb), r0=r0)

    def update_aabb(self, new_aabb):
        self.near = torch.Tensor([0, 0, -pi]).to(self.device)
        self.far = torch.Tensor([self._get_max_r(new_aabb), pi, pi]).to(self.device)
        self.inv_diff = 1.0 / (self.far - self.near)

    def from_cartesian(self, xyz_points):
        """
        convert [x, y, z] to [r, theta, phi]
        xyz_points: (N_rays, N_sample, 3) or (N_rays * N_sample, 3)
        """
        diff = xyz_points - self.center
        r = diff.pow(2).sum(-1).sqrt()
        theta = torch.acos(diff[..., 2] / r).nan_to_num_()
        phi = torch.atan2(diff[..., 1], diff[..., 0])

        return torch.stack([r, theta, phi], dim=-1)

    def normalize_coord(self, unnormalized_coords):
        if not self.exp_r:
            return (unnormalized_coords - self.near) * self.inv_diff * 2 - 1
        else:
            r = unnormalized_coords[..., 0] - self.near[0]
            norm_r = self.normalize_r(r) * 2 - 1
            norm_theta_phi = (unnormalized_coords[..., 1:] - self.near[1:]) * self.inv_diff[1:] * 2 - 1
            return torch.cat((torch.unsqueeze(norm_r, -1), norm_theta_phi), dim=-1)

    def normalize_r(self, r, downsample=None):

        if self.interval_th:
            # normalized_coord = torch.zeros_like(r)
            # N_r = self.N_r if downsample is None else self.N_r // downsample
            # r0 = self.r0 if downsample is None else self.r0 * downsample
            N_r = self.N_r
            r0 = self.r0
            ratio = pow(self.far[0] / r0, 1 / (N_r - 1))
            reference_r_grid = index2r(r0, ratio, torch.arange(N_r + 1).to(self.device))
            reference_interval = reference_r_grid[1:] - reference_r_grid[:-1]
            reference_interval_cum = torch.cumsum(reference_interval, dim=0)
            interval_less_than_r0 = reference_interval <= r0
            reference_r_grid[:interval_less_than_r0.sum() + 1] = torch.arange(interval_less_than_r0.sum() + 1) * r0
            reference_r_grid[interval_less_than_r0.sum() + 1:] = reference_r_grid[interval_less_than_r0.sum() + 1:] + r0 * interval_less_than_r0.sum() - reference_interval_cum[interval_less_than_r0.sum() - 1]
            coords_out = torch.clamp(torch.searchsorted(reference_r_grid, r, side='right'), 1, reference_r_grid.shape[0]-1)
            coords_in = coords_out - 1
            r_coords_out = reference_r_grid[coords_out]
            r_coords_in = reference_r_grid[coords_in]

            linear_space = (r - r_coords_in) / (r_coords_out - r_coords_in)
            normalized_coord = coords_in + linear_space
        else:
            r0 = self.r0
            if downsample is None:
                N_r = self.N_r
                ratio = self.ratio
            else:
                N_r = self.N_r // downsample
                ratio = pow(self.far[0] / r0, 1 / (N_r - 1))

            k = torch.log(r / r0) / log(ratio)
            normalized_coord = torch.zeros_like(k)
            k = k.type(torch.IntTensor).to(self.device)
            is_r_less_than_r0 = r < r0

            r_coords_in = r0 * torch.pow(ratio, k)
            r_coords_in[is_r_less_than_r0] = 0

            r_coords_out = r0 * torch.pow(ratio, k + 1)
            r_coords_out[is_r_less_than_r0] = r0

            linear_space = (r - r_coords_in) / (r_coords_out - r_coords_in)

            normalized_coord[is_r_less_than_r0] = r[is_r_less_than_r0] / r0
            normalized_coord[torch.logical_not(is_r_less_than_r0)] = 1 + k[torch.logical_not(is_r_less_than_r0)] + linear_space[torch.logical_not(is_r_less_than_r0)]
        return normalized_coord / N_r

    """
    def unnormalize_r(self, norm_r, target_reso):
        assert self.exp_r, 'unnormalize_r is only for exponential r grid!'
        # r0 = self.r0 * target_reso / self.N_r
        ratio = exp(log(self.far[0] / self.r0) / (target_reso - 1))
        is_r_less_than_r0 = norm_r < 1
        unnormalized_r = torch.zeros_like(norm_r)
        unnormalized_r[is_r_less_than_r0] = norm_r[is_r_less_than_r0] * self.r0

        norm_r_in = norm_r.type(torch.IntTensor).to(self.device)
        unnormalized_r_in = self.r0 * torch.pow(ratio, norm_r_in - 1)
        unnormalized_r_out = self.r0 * torch.pow(ratio, norm_r_in)

        unnormalized_r[torch.logical_not(is_r_less_than_r0)] = (norm_r[torch.logical_not(is_r_less_than_r0)] - norm_r_in[torch.logical_not(is_r_less_than_r0)]) * (unnormalized_r_out[torch.logical_not(is_r_less_than_r0)] - unnormalized_r_in[torch.logical_not(is_r_less_than_r0)]) + unnormalized_r_in[torch.logical_not(is_r_less_than_r0)]

        return unnormalized_r
    """

    def N_to_reso(self, n_voxels, bbox):
        N_r = int(pow(n_voxels, 1 / 3) / 2)
        N_theta = N_r * 2
        N_phi = N_theta * 2

        N_r = N_r + 1 if N_r % 2 else N_r
        N_theta = N_theta + 1 if N_theta % 2 else N_theta
        N_phi = N_phi + 1 if N_phi % 2 else N_phi

        return [N_r, N_theta, N_phi]

    def _get_max_r(self, aabb):
        """
        get maximum value of radius
        """
        xyz_min, xyz_max = aabb.tolist()

        # return (xyz_max[0] - xyz_min[0]) / 2.0
        corner_points = []
        for i in range(8):
            point = []
            for b in range(3):
                selector = 1 << b
                point.append(xyz_min[b]) if i & selector != 0 else point.append(xyz_max[b])
            corner_points.append(point)

        corner_points = torch.Tensor(corner_points).to(self.device)

        return (corner_points - self.center).pow(2).sum(1).sqrt().amax()

    def set_resolution(self, resolution, r0=None):
        self.N_r = resolution[0]
        self.N_theta = resolution[1]
        self.N_phi = resolution[2]

        if self.exp_r:
            # heuristic r0 choice..
            # k th index's position: r0 * ratio ** k
            self.r0 = r0 if r0 is not None else 0.05
            self.ratio = pow(self.far[0] / self.r0, 1 / (self.N_r - 1))
        """
        if self.interval_th:
            # N_r0: maximum index that grid interval is r0. (i.e. r[N_r0] - r[N_r0 - 1] = r0)
            indices = torch.arange(self.N_r).to(self.device)
            r = index2r(self.r0, self.ratio, indices)
            interval = r[1:] - r[:-1]
            interval_less_than_r0 = interval <= r0
            self.N_r0 = interval_less_than_r0.sum() + 1 if interval_less_than_r0.sum() % 2 else interval_less_than_r0.sum()
        """

    def up_sampling_VM(self, weights, res_target, ids):
        assert len(ids) == 1 or len(ids) == 2, 'ids should be 1 or 2!'
        if not self.exp_r:
            super().up_sampling_VM(weights, res_target, ids)
        else:
            r_idx = next((i for i, id in enumerate(ids) if id == 0), -1)
            if r_idx == -1:
                return super().up_sampling_VM(weights, res_target, ids)
            else:
                # r_samples = self.normalize_r((torch.pow(self.ratio, torch.arange(res_target[0], device=self.device)) - 1) / self.coeff) * 2 - 1
                # unnormalized_coords = self.unnormalize_r(torch.linspace(0, self.N_r, res_target[0], device=self.device), res_target[0])
                # unnormalized_coords = self.unnormalize_r(torch.arange(res_target[0]).type(torch.FloatTensor).to(self.device), res_target[0])
                ratio = pow(self.far[0] / self.r0, 1 / (res_target[0] - 1))
                if self.interval_th:
                    reference_r_grid = index2r(self.r0, ratio, torch.arange(res_target[0], device=self.device))
                    reference_interval = reference_r_grid[1:] - reference_r_grid[:-1]
                    reference_interval_cum = torch.cumsum(reference_interval, dim=0)
                    interval_less_than_r0 = reference_interval <= self.r0
                    reference_r_grid[:interval_less_than_r0.sum() + 1] = torch.arange(interval_less_than_r0.sum() + 1) * self.r0
                    reference_r_grid[interval_less_than_r0.sum() + 1:] = reference_r_grid[interval_less_than_r0.sum() + 1:] + self.r0 * interval_less_than_r0.sum() - reference_interval_cum[interval_less_than_r0.sum() - 1]
                    r_samples = self.normalize_r(reference_r_grid) * 2 - 1
                else:
                    unnormalized_coords = torch.zeros(res_target[0], device=self.device)
                    unnormalized_coords[1:] = self.r0 * (torch.pow(ratio, torch.arange(res_target[0] - 1, device=self.device)))
                    r_samples = self.normalize_r(unnormalized_coords) * 2 - 1
                if len(ids) == 1:
                    grid = torch.stack((-torch.ones_like(r_samples), r_samples)).expand((1, 1, 2, res_target[0])).permute([0, 3, 1, 2])
                    return F.grid_sample(weights, grid, align_corners=True)
                else:
                    other_idx = (r_idx + 1) % 2
                    other_samples = torch.linspace(-1, 1, res_target[ids[other_idx]], device=self.device).expand(1, res_target[0], res_target[ids[other_idx]])
                    r_samples = r_samples.expand(1, res_target[ids[other_idx]], res_target[0])

                    if r_idx == 0:
                        r_samples = r_samples.transpose(1,2)
                    else:
                        other_samples = other_samples.transpose(1,2)

                    grid = torch.stack((other_samples, r_samples), dim=-1)[..., [r_idx, other_idx]]

                    return F.grid_sample(weights, grid, align_corners=True)


class SphericalCoords(Coordinates):
    """
    Spherical Coordinates[r, theta, phi]
    """
    def __init__(self, device, aabb):
        self.center = aabb.to(device).sum(0).div(2)
        super(SphericalCoords, self).__init__(device, aabb)

    def normalize_coord(self, sample_points):
        """
        sampled points:
            un-normalized points. ex([3, pi/2, pi])
        we need to normalize [r, theta, phi] to [-1, 1] range
        """
        return (sample_points - self.near) * self.inv_diff * 2 - 1

    def from_cartesian(self, xyz_points):
        """
        convert [x, y, z] to [r, theta, phi]
        xyz_points: (N_rays, N_sample, 3) or (N_rays * N_sample, 3)
        """
        diff = xyz_points - self.center
        r = diff.pow(2).sum(-1).sqrt()
        theta = torch.acos(diff[..., 2] / r).nan_to_num_()
        phi = torch.atan2(diff[..., 1], diff[..., 0])

        return torch.stack([r, theta, phi], dim=-1)

    def update_aabb(self, new_aabb):
        self.near = torch.Tensor([0, 0, -pi]).to(self.device)
        self.far = torch.Tensor([self._get_max_r(new_aabb), pi, pi]).to(self.device)
        self.inv_diff = 1.0 / (self.far - self.near)

    def get_normalized_range(self, new_aabb):
        """
        Spherical coordinates need to shrink while maintaining spherical shape; i.e. only r need to be affected
        """
        norm_r_max = (self._get_max_r(new_aabb) - self.near[0]) * self.inv_diff[0]
        return torch.Tensor([0, 0, 0]).to(self.device), torch.Tensor([norm_r_max, 1, 1]).to(self.device)

    def N_to_reso(self, n_voxels, bbox):
        # TODO
        N_r = int(pow(n_voxels, 1 / 3) / 2)
        N_theta = N_r * 2
        N_phi = N_theta * 2
        # xyz_min, xyz_max = bbox
        # dim = len(xyz_min)
        # return [int(pow(n_voxels, 1 / dim))] * 3
        return [N_r, N_theta, N_phi]

    def _get_max_r(self, aabb):
        """
        get maximum value of radius
        """
        xyz_min, xyz_max = aabb.tolist()

        corner_points = []
        for i in range(8):
            point = []
            for b in range(3):
                selector = 1 << b
                point.append(xyz_min[b]) if i & selector != 0 else point.append(xyz_max[b])
            corner_points.append(point)
        
        corner_points = torch.Tensor(corner_points).to(self.device)

        return (corner_points - self.center).pow(2).sum(1).sqrt().amax()


class DirectionalSphericalCoords(SphericalCoords):
    """
    Directional Spherical Coordinates[r, theta, phi]
    """
    def normalize_coord(self, sample_points):
        """
        sampled points:
            un-normalized points. ex([3, pi/2, pi])
        we need to normalize [r, theta, phi] to [-1, 1] range
        """
        r = sample_points[..., 0]
        theta = sample_points[..., 1]
        phi = sample_points[..., 2]

        r[phi < 0] *= -1
        theta[phi < 0] *= -1
        phi[phi < 0] += pi

        r = r * self.inv_diff[0]
        theta = theta * self.inv_diff[1]
        phi = phi * self.inv_diff[2] * 2 - 1

        return torch.stack([r, theta, phi], dim=-1)

    def update_aabb(self, new_aabb):
        # note that theta, phi have same range [0, pi]
        self.near = torch.Tensor([0, 0, 0]).to(self.device)
        self.far = torch.Tensor([self._get_max_r(new_aabb), pi, pi]).to(self.device)
        self.inv_diff = 1.0 / (self.far - self.near)


class BalancedSphericalCoords(SphericalCoords):
    """
    Balanced Spherical Coordinates[r, theta, phi]
    """
    def normalize_coord(self, sample_points):
        # normalize r to maintain balanced spherical frustum
        r = sample_points[..., 0] - self.near[0]
        norm_r = self.normalize_r(r) * 2 - 1
        norm_theta_phi = (sample_points[..., 1:] - self.near[1:]) * self.inv_diff[1:] * 2 - 1

        return torch.cat((torch.unsqueeze(norm_r, -1), norm_theta_phi), dim=-1)

    def normalize_r(self, r):
        # linear interpolation between nearest coordinate indicies
        k = (torch.log((r * self.coeff) + 1) / log(self.ratio))
        k = k.type(torch.IntTensor).to(self.device)
        # TODO: cache coordinates r values
        r_coords_in = (torch.pow(self.ratio, k) - 1) / self.coeff
        r_coords_out = (torch.pow(self.ratio, k + 1) - 1) / self.coeff

        linear_space = (r - r_coords_in) / (r_coords_out - r_coords_in)

        return ((k + linear_space) / self.resolution[0])

    def N_to_reso(self, n_voxels, bbox):
        # TODO: find the best ratio of N_r, N_theta, N_phi
        #   might be dependent to scene configuration
        #   current setting: N_r=N_theta=N_phi/2
        N_r = int(pow(n_voxels, 1/3) / 2)
        N_theta = N_r * 2
        N_phi = N_theta * 2

        self.ratio = 1 + pi / N_theta
        self.r0 = (self.ratio - 1) / pow(self.ratio, N_r) * self.far[0]
        self.coeff = (self.ratio - 1) / self.r0

        return [N_r, N_theta, N_phi]

    def up_sampling_VM(self, weights, res_target, ids):
        assert len(ids) == 1 or len(ids) == 2, 'ids should be 1 or 2!'
        r_idx = next((i for i, id in enumerate(ids) if id == 0), -1)
        if r_idx == -1:
            return super().up_sampling_VM(weights, res_target, ids)
        else:
            r_samples = self.normalize_r((torch.pow(self.ratio, torch.arange(res_target[0], device=self.device)) - 1) / self.coeff) * 2 - 1
            if len(ids) == 1:
                grid = torch.stack((-torch.ones_like(r_samples), r_samples)).expand((1, 1, 2, res_target[0])).permute([0, 3, 1, 2])
                return F.grid_sample(weights, grid, align_corners=True)
            else:
                other_idx = (r_idx + 1) % 2
                other_samples = torch.linspace(-1, 1, res_target[ids[other_idx]], device=self.device).expand(1, res_target[0], res_target[ids[other_idx]])
                r_samples = r_samples.expand(1, res_target[ids[other_idx]], res_target[0])

                if r_idx == 0:
                    r_samples = r_samples.transpose(1,2)
                else:
                    other_samples = other_samples.transpose(1,2)

                grid = torch.stack((other_samples, r_samples), dim=-1)[..., [r_idx, other_idx]]

                return F.grid_sample(weights, grid, align_corners=True)


class YinYangSphericalCoords(GenericSphericalCoords):
    """
    Yin-Yang Spherical Coordinates [r_n, theta_n, phi_n, r_e, theta_e, phi_e, Y]
        r_n, theta_n, phi_n: Yin grid
        r_e, theta_e, phi_e: Yang grid
        Y: 0 -> Yin grid, 1 -> Yang grid
    """
    def __init__(self, device, aabb, exp_r=True, N_voxel=None, r0=None, interval_th=False):
        super(YinYangSphericalCoords, self).__init__(device, aabb, exp_r=exp_r, N_voxel=N_voxel, r0=r0, interval_th=interval_th)

    def normalize_coord(self, unnormalized_coords, downsample=None):
        """
        unnormalized_coords:
            un-normalized coordinates. ex([3, pi/4, pi/8, 3, 0, 0, 0])
        We need to normalize [r_n, theta_n, phi_n, r_e, theta_e, phi_e, Y] to [-1, 1] range except Y
        """
        if not self.exp_r:
            normalized_coords = torch.zeros_like(unnormalized_coords)
            normalized_coords[..., :-1] = (unnormalized_coords[..., :-1] - self.near) * self.inv_diff * 2 - 1
            normalized_coords[..., -1] = unnormalized_coords[..., -1]
        else:
            r_yin = unnormalized_coords[..., 0] - self.near[0]
            r_yang = unnormalized_coords[..., 3] - self.near[0]
            norm_r_yin = self.normalize_r(r_yin, downsample=downsample) * 2 - 1
            norm_r_yang = self.normalize_r(r_yang, downsample=downsample) * 2 - 1
            norm_theta_phi_yin = (unnormalized_coords[..., 1:3] - self.near[1:3]) * self.inv_diff[1:3] * 2 - 1
            norm_theta_phi_yang = (unnormalized_coords[..., 4:6] - self.near[4:6]) * self.inv_diff[4:6] * 2 - 1
            normalized_coords = torch.cat((
                torch.unsqueeze(norm_r_yin, -1),
                norm_theta_phi_yin,
                torch.unsqueeze(norm_r_yang, -1),
                norm_theta_phi_yang,
                torch.unsqueeze(unnormalized_coords[..., -1], -1)
            ), dim=-1)
        return normalized_coords

    def from_cartesian(self, xyz_points):
        """
        convert [x, y, z] to [r_n, theta_n, phi_n, r_e, theta_e, phi_e, Y]
        xyz_points: (N_rays, N_samples, 3) or (N_rays * N_sample, 3)
        """
        coords = torch.zeros(*xyz_points.shape[:-1], 7, device=xyz_points.device)

        diff = xyz_points - self.center
        r = diff.pow(2).sum(-1).sqrt()
        theta_n = torch.acos(diff[..., 2] / r).nan_to_num_()
        phi_n = torch.atan2(diff[..., 1], diff[..., 0])

        yin_grid = torch.stack((r, theta_n, phi_n), dim=-1)
        yin_grid = torch.cat((yin_grid, torch.zeros(*coords.shape[:-1], 4, device=yin_grid.device)), -1)

        yin_filter = torch.logical_and(
            torch.logical_and(pi / 4 <= theta_n, theta_n <= 3 * pi / 4),
            torch.logical_and(-3 * pi / 4 <= phi_n, phi_n <= 3 * pi / 4)
        )
        coords[yin_filter] = yin_grid[yin_filter]

        yang_filter = torch.logical_not(yin_filter)

        theta_e = torch.acos(diff[..., 1] / r).nan_to_num_()
        phi_e = torch.atan2(diff[..., 2], -diff[..., 0])

        yang_grid = torch.stack((r, theta_e, phi_e, torch.ones_like(r)), dim=-1)
        yang_grid = torch.cat((torch.zeros(*coords.shape[:-1], 3, device=yang_grid.device), yang_grid), -1)
        coords[yang_filter] = yang_grid[yang_filter]

        return coords

    def update_aabb(self, new_aabb):
        self.near = torch.Tensor([0, pi / 4, -3 * pi / 4, 0, pi / 4, -3 * pi / 4]).to(self.device)
        self.far = torch.Tensor([
            self._get_max_r(new_aabb), 3 * pi / 4, 3 * pi / 4, self._get_max_r(new_aabb), 3 * pi / 4, 3 * pi / 4
        ]).to(self.device)
        self.inv_diff = 1.0 / (self.far - self.near)

    def N_to_reso(self, n_voxels, bbox):
        # TODO: find the best ratio of N_r and N_theta. N_phi is fixed to N_theta * 3
        # Yin and Yang grid have same N_r, N_theta, N_phi values.
        # Current setting: N_q1 : N_q2 : N_q3 = 1 : 2 * sqrt(3) / 3 : 2 * sqrt(3)
        #                  N_r * N_theta * N_phi = n_voxels / 2
        N_r = int(pow(n_voxels, 1 / 3) / 2)
        N_theta = int(N_r * 2 * sqrt(3) / 3)
        N_phi = N_theta * 3

        N_r = N_r + 1 if N_r % 2 else N_r
        N_theta = N_theta + 1 if N_theta % 2 else N_theta
        N_phi = N_phi + 1 if N_phi % 2 else N_phi

        return [N_r, N_theta, N_phi]


# TODO: Merge spherical coordinates together, extract the features as attributes
class DirectionalBalancedSphericalCoords(BalancedSphericalCoords):
    """
    Spherical Coordinates satisfy both conditions 'directional' & 'balanced'
    """
    def normalize_coord(self, sample_points):
        # normalize r to maintain balanced spherical frustum
        r = sample_points[..., 0] - self.near[0]
        norm_r = self.normalize_r(r)
        theta = sample_points[..., 1]
        phi = sample_points[..., 2]

        norm_r[phi < 0] *= -1
        theta[phi < 0] *= -1
        phi[phi < 0] += pi

        theta = theta * self.inv_diff[1]
        phi = phi * self.inv_diff[2] * 2 - 1

        return torch.stack([norm_r, theta, phi], dim=-1)

    def update_aabb(self, new_aabb):
        # note that theta, phi have same range [0, pi]
        self.near = torch.Tensor([0, 0, 0]).to(self.device)
        self.far = torch.Tensor([self._get_max_r(new_aabb), pi, pi]).to(self.device)
        self.inv_diff = 1.0/(self.far - self.near)

    def N_to_reso(self, n_voxels, bbox):
        # TODO: find the best ratio of N_r, N_theta, N_phi
        #   mighe be dependent to scene configuration
        #   current setting: N_r=N_theta=N_phi
        N_r = int(pow(n_voxels, 1/3))
        N_theta = N_r
        N_phi = N_theta

        self.ratio = 1 + pi / N_theta
        self.r0 = (self.ratio - 1) / pow(self.ratio, N_r // 2) * self.far[0]
        self.coeff = (self.ratio - 1) / self.r0

        return [N_r, N_theta, N_phi]

    def up_sampling_VM(self, weights, res_target, ids):
        '''
        exponential sampling for r
        '''
        # TODO: refactoring messy code
        assert len(ids) == 1 or len(ids) == 2, 'len(ids) should be 1 or 2!'

        r_idx = next((i for i, id in enumerate(ids) if id==0), -1)

        if r_idx == -1:
            return super().up_sampling_VM(weights, res_target, ids)
        else:
            if res_target[0] % 2 == 0:
                one_dir_sample_cnt = res_target[0] // 2
                one_dir_samples = self.normalize_r((torch.pow(self.ratio, torch.arange(one_dir_sample_cnt, device=self.device)) - 1) / self.coeff)
                r_samples = torch.cat([-one_dir_samples.flip(0), one_dir_samples])
            else:
                one_dir_sample_cnt = res_target[0] // 2 + 1
                one_dir_samples = self.normalize_r((torch.pow(self.ratio, torch.arange(one_dir_sample_cnt, device=self.device)) - 1) / self.coeff)
                r_samples = torch.cat([-one_dir_samples.flip(0)[:-1], one_dir_samples])

            if len(ids) == 1:
                grid = torch.stack((r_samples, torch.zeros(res_target[0], device=self.device))).expand((1,1,2,res_target[0])).permute([0,3,1,2])
                return F.grid_sample(weights, grid, align_corners=True)
            else:
                other_idx = (r_idx + 1) % 2
                other_samples = torch.linspace(-1, 1, res_target[ids[other_idx]], device=self.device).expand(1, res_target[0], res_target[ids[other_idx]])
                r_samples = r_samples.expand(1, res_target[ids[other_idx]], res_target[0])

                if r_idx == 0:
                    r_samples = r_samples.transpose(1,2)
                else:
                    other_samples = other_samples.transpose(1,2)

                grid = torch.stack((r_samples, other_samples), dim=-1)[..., [r_idx, other_idx]]

                return F.grid_sample(weights, grid, align_corners=True)

    def set_resolution(self, resolution):
        resolution[0] //= 2
        self.resolution = resolution

        
class EulerSphericalCoords(Coordinates):
    '''
    Euler Spherical Coordinates[r, pitch, yaw]
    '''
    def __init__(self, device, aabb):
        self.center = aabb.to(device).sum(0).div(2)
        super(EulerSphericalCoords, self).__init__(device, aabb)

    def normalize_coord(self, sample_points):
        '''
        sampled points:
            un-normalized points. ex([3, pi/2, pi])
        we need to normalize [r, theta, phi] to [-1, 1] range
        '''
        return (sample_points-self.near) * self.inv_diff * 2 - 1

    def from_cartesian(self, xyz_points):
        '''
        convert [x, y, z] to [r, theta, phi]
        xyz_points: (N_rays, N_sample, 3) or (N_rays * N_sample, 3)
        '''
        diff = xyz_points - self.center
        r = diff.pow(2).sum(-1).sqrt()
        pitch = torch.atan2(diff[...,2], diff[...,0])
        yaw = torch.atan2(diff[...,2], diff[...,1])

        return torch.stack([r, pitch, yaw], dim=-1)

    def update_aabb(self, new_aabb):
        self.near = torch.Tensor([0, -pi, -pi]).to(self.device)
        self.far = torch.Tensor([self._get_max_r(new_aabb), pi, pi]).to(self.device)
        self.inv_diff = 1.0/(self.far - self.near)

    def get_normalized_range(self, new_aabb):
        '''
        Spherical coordinates need to shrink while maintaining spherical shape; i.e. only r need to be affected
        '''
        norm_r_max = (self._get_max_r(new_aabb) - self.near[0]) * self.inv_diff[0]
        return torch.Tensor([0,0,0]).to(self.device), torch.Tensor([norm_r_max,1,1]).to(self.device)

    def N_to_reso(self, n_voxels, bbox):
        N_r = int(pow(n_voxels, 1/3) / 2)
        N_pitch = N_yaw = int(N_r * 2 * sqrt(2))
        # N_r = int(pow(n_voxels, 1/3))
        # N_pitch = N_yaw = N_r
        return [N_r, N_pitch, N_yaw]

    def _get_max_r(self, aabb):
        '''
        get maximum value of radius
        '''
        xyz_min, xyz_max = aabb.tolist()

        corner_points = []
        for i in range(8):
            point = []
            for b in range(3):
                selector = 1 << b
                point.append(xyz_min[b]) if i & selector != 0 else point.append(xyz_max[b])
            corner_points.append(point)
        
        corner_points = torch.Tensor(corner_points).to(self.device)

        return (corner_points - self.center).pow(2).sum(1).sqrt().amax()


class CylindricalCoords(Coordinates):
    """
    Cylindrical Coordinates[rho, phi, z]
    """
    def __init__(self, device, aabb):
        self.center = aabb.to(device).sum(0).div(2)
        super(CylindricalCoords, self).__init__(device, aabb)

    def update_aabb(self, new_aabb):
        aabb = new_aabb.to(self.device)
        self.near = torch.Tensor([0, -pi, aabb[0][2]]).to(self.device)
        self.far = torch.Tensor([(aabb[1][:2] - self.center[:2]).max(), pi, aabb[1][2]]).to(self.device) # set r_far as the farthest normal distance from center to bounding box
        self.inv_diff = 1.0/(self.far - self.near)

    def from_cartesian(self, xyz_points):
        """
        convert [x, y, z] to [rho, phi, z]
        xyz_points: (N_rays, N_sample, 3) or (N_rays * N_sample, 3)
        """
        diff = xyz_points[..., :2] - self.center[:2]  # (N_rays, N_sample, 3)
        rho = diff.pow(2).sum(-1).sqrt()  # (N_rays, N_sample)
        phi = torch.atan2(diff[..., 1], diff[..., 0])
        z = xyz_points[..., 2]

        return torch.stack([rho, phi, z], dim=-1)  # (N_rays, N_sample, 3)

    def normalize_coord(self, sample_points):
        return (sample_points - self.near) * self.inv_diff * 2 - 1

    def get_normalized_range(self, new_aabb):
        norm_rho_max = (self._get_max_rho(new_aabb) - self.near[0]) * self.inv_diff[0]
        norm_z = (new_aabb[:, 2] - self.near[2]) * self.inv_diff[2]
        
        return torch.Tensor([0, 0, norm_z[0]]).to(self.device), torch.Tensor([norm_rho_max, 1, norm_z[1]]).to(self.device)

    def N_to_reso(self, n_voxels, bbox):
        # TODO
        xyz_min, xyz_max = bbox
        dim = len(xyz_min)
        return [int(pow(n_voxels, 1 / dim))] * 3

    def _get_max_rho(self, aabb):
        """
        get maximum value of radius
        """
        xyz_min, xyz_max = aabb.tolist()

        corner_points = []
        for i in range(4):
            point = []
            for b in range(2):
                selector = 1 << b
                point.append(xyz_min[b]) if i & selector != 0 else point.append(xyz_max[b])
            corner_points.append(point)
        
        corner_points = torch.Tensor(corner_points).to(self.device)

        return (corner_points - self.center[:2]).pow(2).sum(1).sqrt().amax()
