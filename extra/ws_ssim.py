import numpy as np
import cv2
import numpy
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi
from torchmetrics import StructuralSimilarityIndexMeasure


myfloat = np.float64

def generate_ws(i,j,M,N):
    res = np.cos( (i+0.5-N/2)*np.pi/N )
    return res

def estws(map_ssim):
    N, M = map_ssim.shape
    ws_map = np.zeros_like(map_ssim)

    for i in range(N):
        for j in range(M):
            ws_map[i][j] = generate_ws(i,j,M,N)

    return ws_map

def ws_ssim(image1, image2):
    ssim, map_ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True)(image1, image2)
    
    map_ssim = map_ssim.squeeze().mean(0).numpy()
    ws = estws(map_ssim)
    wsssim = np.sum(map_ssim * ws) / ws.sum()

    return ssim, wsssim
