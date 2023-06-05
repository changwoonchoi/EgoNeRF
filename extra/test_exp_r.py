import matplotlib.pyplot as plt
from math import exp, log
import torch

N_r = 150
far_list = [15, 50, 500]
r0 = 0.001


def index2r(r0, ratio, index: torch.Tensor):
    non_zero_idx = index > 0
    r = torch.zeros_like(index, dtype=torch.float32)
    r[non_zero_idx] = r0 * ratio ** (index[non_zero_idx] - 1)
    r[~non_zero_idx] = 0
    return r


def test_exp_r():
    for far in far_list:
        indices = torch.arange(N_r + 1)
        ratio = exp(log(far / r0) / (N_r - 1))
        r = index2r(r0, ratio, indices)
        interval = r[1:] - r[:-1]
        interval_cum = torch.cumsum(interval, dim=0)
        interval_less_than_r0 = interval <= r0
        r[:interval_less_than_r0.sum() + 1] = torch.arange(interval_less_than_r0.sum() + 1) * r0
        r[interval_less_than_r0.sum() + 1:] = r[interval_less_than_r0.sum() + 1:] + r0 * interval_less_than_r0.sum() - interval_cum[interval_less_than_r0.sum() - 1]
        print(f"N_r: {N_r}, far: {far}, r0: {r0}, constant_index: {interval_less_than_r0.sum()}")
        print(r)


if __name__ == "__main__":
    test_exp_r()
