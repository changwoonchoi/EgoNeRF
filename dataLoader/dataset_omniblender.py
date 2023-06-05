import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from dataLoader.dataset_interface import EgoNeRFDataset

from .ray_utils import *


class OmniBlenderDataset(EgoNeRFDataset):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		self.img_wh = (int(2000 / self.downsample), int(1000 / self.downsample))
		self.define_transforms()

		self.read_meta()

		self.scene_bbox = self.get_scene_bbox()

		self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

	def get_scene_bbox(self):
		cam_position = self.poses[:, :3, 3]
		self.center = cam_position.mean(0)
		trajectory_radius = (cam_position.max(0).values - cam_position.min(0).values).pow(2).sum(0).sqrt().div(
			2).float()
		return torch.stack([
			self.center - trajectory_radius - self.near_far[1],
			self.center + trajectory_radius + self.near_far[1]
		])

	def read_meta(self):
		with open(os.path.join(self.root_dir, "transform.json"), 'r') as f:
			self.meta = json.load(f)

		w, h = self.img_wh
		self.indoor = self.meta['indoor']

		# ray directions for all pixels, same for all images (same H, W, focal)
		self.directions = get_ray_directions_360(h, w)  # (h, w, 3)
		self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

		# read image list from train/test split file
		if self.split == "train" or self.split == "test":
			img_list_file = open(os.path.join(self.root_dir, f"{self.split}.txt"))
			while True:
				line = img_list_file.readline()
				if not line:
					break
				self.img_list.append(line.strip())
		else:
			raise ValueError("Unknown split: {}".format(self.split))

		if self.split == "train":
			assert self.skip == 1, "skip must be 1 for training"
		self.img_list = self.img_list[::self.skip]

		frame_img_list = [self.meta['frames'][i]['file_path'].split(".")[0] for i in range(len(self.meta['frames']))]
		for img_name in tqdm(self.img_list, desc=f'Loading data {self.split} ({len(self.img_list)})'):
			i = frame_img_list.index(img_name)
			frame = self.meta['frames'][i]
			pose = np.array(frame['transform_matrix']) @ self.blender2opencv
			c2w = torch.FloatTensor(pose)

			self.poses += [c2w]

			image_path = os.path.join(self.root_dir, 'images', f"{frame['file_path']}")
			self.image_paths += [image_path]
			img = Image.open(image_path)

			if self.downsample != 1.0:
				img = img.resize(self.img_wh, Image.LANCZOS)
			img = self.transform(img)  # (4, h, w)
			if img.shape[0] == 3:
				img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
			elif img.shape[0] == 4:
				img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
				img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
			self.all_rgbs += [img]

			rays_o, rays_d = get_rays(self.directions, c2w, self.roi)  # both (h*w, 3)
			self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

		self.poses = torch.stack(self.poses)
		if not self.is_stack:
			self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
			self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

		#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
		else:
			self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
			self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
		# self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

	def define_transforms(self):
		self.transform = T.ToTensor()

	def world2ndc(self, points, lindisp=None):
		device = points.device
		return (points - self.center.to(device)) / self.radius.to(device)

	def __len__(self):
		return len(self.all_rgbs)

	def __getitem__(self, idx):
		if self.split == 'train':  # use data in the buffers
			sample = {
				'rays': self.all_rays[idx],
				'rgbs': self.all_rgbs[idx]
			}
		else:  # create data for each image separately
			img = self.all_rgbs[idx]
			rays = self.all_rays[idx]
			mask = self.all_masks[idx]  # for quantity evaluation
			sample = {
				'rays': rays,
				'rgbs': img,
				'mask': mask
			}
		return sample
