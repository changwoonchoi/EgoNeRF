import torch
import torch.nn.functional as F
from math import pi


def direction_to_canonical(direction):
    direction = F.normalize(direction, dim=-1)
    cos_theta = direction[:, 2]
    phi = torch.atan2(direction[:, 1], direction[:, 0])
    phi += pi
    u = (cos_theta + 1) * 0.5
    v = phi / (2 * pi)
    uv = torch.stack([u, v], dim=1)
    return uv


class EnvironmentMap:
    def __init__(self, h=1000, init_strategy="random", device="cuda"):
        if init_strategy == "random":
            self.emission = torch.rand((3, 2 * h, h), requires_grad=True, device=device)
        elif init_strategy == "zero":
            self.emission = torch.zeros((3, 2 * h, h), requires_grad=True, device=device)
        else:
            raise ValueError("Unknown environment map initialization: {}".format(init_strategy))

    def get_radiance(self, direction):
        # TODO: need to force envmap to output values between 0 and 1?
        uv = direction_to_canonical(direction)
        uv = 2 * uv - 1
        env_radiance = F.grid_sample(self.emission[None, ...], uv[None, :, None, ...], align_corners=True)
        env_radiance = env_radiance.permute((0, 2, 3, 1))
        env_radiance = env_radiance.squeeze()
        env_radiance = torch.sigmoid(env_radiance)  # force output to be between 0 and 1
        return env_radiance

    def load_envmap(self, emission, device):
        self.emission = torch.tensor(emission, requires_grad=True, device=device)
