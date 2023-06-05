from abc import ABC
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataLoader.pose_descriptor import pose_descriptor_dict


class EgoNeRFDataset(Dataset, ABC):
	def __init__(self, **kwargs):
		self.root_dir = kwargs.get("data_dir")
		self.split = kwargs.get("split", "train")
		self.is_stack = kwargs.get("is_stack", False)
		self.use_gt_depth = kwargs.get("use_gt_depth", False)

		self.N_vis = kwargs.get("N_vis", -1)

		self.pose_descriptor = pose_descriptor_dict[kwargs.get("localization_method", "colmap")]()

		self.roi = kwargs.get("roi", [0, 1, 0, 1])
		self.blender2opencv = np.eye(4)

		self.meta = None

		self.indoor = True
		self.white_bg = False
		self.near_far = kwargs.get("near_far", [0.1, 15.0])
		self.scene_bbox = None

		self.center = None
		self.radius = None
		self.downsample = kwargs.get("downsample", 1.0)

		self.img_wh = (0, 0)
		self.focal = None
		self.directions = None
		self.intrinsics = None
		self.image_paths = []
		self.poses = []
		self.all_rays = []
		self.all_rgbs = []
		self.all_masks = []
		self.all_depth = []

		self.img_list = []
		self.skip = kwargs.get("skip", 1)
		self.transform = None
		self.proj_mat = None

	def __getitem__(self, idx):
		pass

	def __len__(self):
		pass
