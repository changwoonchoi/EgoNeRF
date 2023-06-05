import torch
import numpy as np

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


class ThetaImportanceSampler:
    def __init__(self, theta_importance_lambda, img_len, img_wh, batch, roi):
        self.img_len = img_len
        self.batch = batch
        W, H = img_wh
        self.W = int(W * (roi[3] - roi[2]))
        self.H = int(H * (roi[1] - roi[0]))
        self.weight = self.get_weight(theta_importance_lambda, H, roi)

    def get_weight(self, theta_importance_lambda, h, roi):
        theta = -(np.arange(h)[int(h*roi[0]):int(h*roi[1])] - h//2) / h * np.pi
        weight = np.cos(theta) * theta_importance_lambda + 1
        weight /= np.sum(weight)
        return weight

    def nextids(self):
        img_id = np.random.choice(self.img_len, self.batch)
        random_width = np.random.choice(self.W, self.batch)
        random_height = np.random.choice(self.H, self.batch, p=self.weight)
        return img_id * self.W * self.H + (random_width + random_height * self.W)