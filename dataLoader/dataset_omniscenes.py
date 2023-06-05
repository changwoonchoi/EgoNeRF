import torch,cv2
import os
from PIL import Image
from torchvision import transforms as T

from dataLoader.dataset_interface import EgoNeRFDataset

from .ray_utils import *


class OmniscenesDataset(EgoNeRFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roi = [0, 0.9, 0, 1]
        self.img_wh_origin = (int(1920 / self.downsample), int(960 / self.downsample))
        self.img_wh = (int(self.img_wh_origin[0] * (self.roi[3] - self.roi[2])), int(self.img_wh_origin[1] * (self.roi[1] - self.roi[0])))
        self.define_transforms()

        # cam: [x,y,z], [forward, left, up]
        # rays: [x,y,z], [right, up, backward]
        self.rays2cam = np.array([
            [0., 0., -1., 0.], 
            [-1., 0, 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            ])
        self.read_meta()

        self.white_bg = False
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
        
        room_name = self.root_dir.split('/')[-1]
        data_dir = '/'.join(self.root_dir.split('/')[:-1])

        img_dir = os.path.join(data_dir, 'turtlebot_pano', room_name)
        pose_dir = os.path.join(data_dir, 'turtlebot_pose', room_name)
        img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir), key=lambda fname: int(fname.split('.')[0][4:])) if f.endswith('.jpg')]
        pose_files = [os.path.join(pose_dir, f) for f in sorted(os.listdir(pose_dir), key=lambda fname: int(fname.split('.')[0][4:])) if f.endswith('.txt')]
        
        assert len(img_files) == len(pose_files)

        if self.split == 'train':
            img_files = img_files[-31:-1]
            pose_files = pose_files[-31:-1]
        elif self.split == 'test':
            img_files = img_files[-1:]
            pose_files = pose_files[-1:]

        w, h = self.img_wh_origin

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_360(h, w)  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        self.all_rays = []
        self.all_rgbs = []
        self.poses = []
        
        for i in range(len(img_files)):
            # load pose
            c2w = torch.FloatTensor(self._load_pose(pose_files[i]))
            self.poses += [c2w]

            # load image
            img = Image.open(img_files[i])
            if self.downsample != 1.0:
                img = img.resize(self.img_wh_origin, Image.LANCZOS)
            img = self.transform(img)
            img = img[:, int(self.roi[0] * h):int(self.roi[1] * h), int(self.roi[2] * w):int(self.roi[3] * w)]
            img = img.view(img.shape[0], -1).permute(1, 0)
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            self.all_rgbs += [img]
            
            # generate rays
            rays_o, rays_d = get_rays(self.directions, c2w, self.roi)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def _load_pose(self, filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        c2w = np.array(nums).astype(np.float32).reshape(3,4)
        c2w[:3, :3] = np.linalg.inv(c2w[:3,:3])
        return c2w @ self.rays2cam
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]

            sample = {'rays': rays,
                      'rgbs': img}
            
        return sample
