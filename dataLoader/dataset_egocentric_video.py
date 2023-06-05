import torch,cv2
from tqdm import tqdm
import os
import glob
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from PIL import Image

from dataLoader.dataset_interface import EgoNeRFDataset
from .ray_utils import *


class EgocentricVideoDataset(EgoNeRFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_wh_origin = (int(1920 / self.downsample), int(960 / self.downsample))  # for Ricoh dataset
        self.img_wh = (int(self.img_wh_origin[0] * (self.roi[3] - self.roi[2])), int(self.img_wh_origin[1] * (self.roi[1] - self.roi[0])))
        self.define_transforms()

        self.blender2opencv = np.eye(4) # np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.white_bg = False
        self.read_meta()
        self.define_proj_mat()

        self.scene_bbox = self.get_scene_bbox()
        
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def get_scene_bbox(self):
        cam_position = self.poses[:, :3, 3]
        self.center = cam_position.mean(0)
        trajectory_radius = (cam_position.max(0).values - cam_position.min(0).values).pow(2).sum(0).sqrt().div(2).float()
        return torch.stack([
            self.center - trajectory_radius - self.near_far[1], 
            self.center + trajectory_radius + self.near_far[1]
        ])

    def read_meta(self):
        img_dir = os.path.join(self.root_dir, 'imgs')

        all_img_files = os.listdir(img_dir)
        if 'mask.png' in all_img_files:
            all_img_files.remove('mask.png')
        # img_files = [os.path.join(img_dir, f) for f in sorted(all_img_files, key=lambda fname: int(fname.split('.')[0])) if f.endswith('.png')]

        if self.split == "train" or self.split == "test":
            img_list_file = open(os.path.join(self.root_dir, f"{self.split}.txt"))
            while True:
                line = img_list_file.readline()
                if not line:
                    break
                self.img_list.append(os.path.join(img_dir, line.strip() + '.png'))
        else:
            raise ValueError("Unknown split: {}".format(self.split))

        w, h = self.img_wh_origin

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_360(h, w)  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        
        self.pose_descriptor.read_pose_file(self.root_dir, img_ext='.png')
        self.pose_descriptor.normalize_pose()

        self.all_rays = []
        self.all_rgbs = []
        self.poses = []

        # for img_fname in tqdm(img_files, desc=f'Loading data {self.split} ({len(img_files)})'):
        for img_fname in tqdm(self.img_list, desc=f'Loading data {self.split} ({len(self.img_list)})'):

            # read image
            img = Image.open(img_fname)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh_origin, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img[:, int(self.roi[0] * h):int(self.roi[1] * h), int(self.roi[2] * w):int(self.roi[3] * w)]
            img = img.reshape(img.shape[0], -1).permute(1, 0)  # (h*w, 3) RGB
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs.append(img)

            c2w = self.pose_descriptor.poses_dict[os.path.basename(img_fname)] @ self.blender2opencv
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)  # C2W

            # rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = get_rays(self.directions, c2w, roi=self.roi)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)

        if 'train' == self.split:
            if self.is_stack:
                self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames])*h*w, 6)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3) 
            else:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 6)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 6)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def rotmat(self, a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]  # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
