import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class PoseDescriptorBase:
    def __init__(self):
        self.poses_dict = {}

    @property
    def rays2cam(self):
        return np.eye(4)   

    @property
    def world_align(self):
        return np.eye(4)
    
    def read_pose_file(self, root_dir, sub_path=None, img_ext=None):
        pass

    def normalize_pose(self):
        '''
        normalize pose s.t. r=1, center=(0,0)
        '''
        if not self.poses_dict:
            print('poses_dict is empty!')
            return
        
        cam_center = np.zeros(3)
        for pose in self.poses_dict.values():
            cam_center += pose[:3, 3]
        cam_center /= len(self.poses_dict)
        
        dist = 0
        for pose in self.poses_dict.values():
            dist += np.sqrt(((pose[:3, 3] - cam_center) ** 2).sum())

        dist /= len(self.poses_dict)

        for pose in self.poses_dict.values():
            pose[:3, 3] = (pose[:3, 3] - cam_center) / dist


class ColmapPoseDescriptor(PoseDescriptorBase):
    def __init__(self):
        super().__init__()
    
    @property
    def rays2cam(self):
        return np.array([
            [1., 0., 0., 0.],
            [0.,-1., 0., 0.],
            [0., 0.,-1., 0.],
            [0., 0., 0., 1.]
            ])

    @property
    def world_align(self):
        return np.array([
            [1., 0., 0., 0.], 
            [0., 0., 1., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 1.]
            ])

    def read_pose_file(self, root_dir, sub_path=None, img_ext=None):
        if not sub_path:
            sub_path = os.path.join('output_dir', 'colmap', 'images.txt')
        pose_file_path = os.path.join(root_dir, sub_path)

        i = 0
        with open(pose_file_path) as f:
            lines = f.readlines()[4:] # discard file info
                
            for line in lines:
                tokens = line.split()
                if tokens[0] == '#':
                    continue
                i += 1
                if i % 2 == 0:
                    continue

                quat, t, img_fname = np.array(list(map(float, tokens[1:5]))), np.array(list(map(float, tokens[5:8]))), tokens[9]
                quat = quat[[1, 2, 3, 0]]
                if img_ext:
                    img_fname = img_fname.split('.')[0] + img_ext

                rot = R.from_quat(quat).as_matrix()
                w2c = np.concatenate((rot, t[:,np.newaxis]), axis=1)
                w2c = np.concatenate((w2c, [[0,0,0,1]]), axis=0)
                c2w = np.linalg.inv(w2c)

                self.poses_dict[img_fname] = self.world_align @ c2w @ self.rays2cam


class OpenVSlamPoseDescriptor(PoseDescriptorBase):
    def __init__(self):
        super().__init__()

    @property
    def rays2cam(self):
        return np.array([
            [0., 0., -1., 0.], 
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]
            ])

    @property
    def world_align(self):
        return np.array([
            [0., 1., 0., 0.], 
            [0., 0., 1., 0.],
            [-1., 0., 0., 0.],
            [0., 0., 0., 1.]
            ])

    def read_pose_file(self, root_dir, sub_path=None, img_ext=None):
        if not sub_path:
            sub_path = os.path.join('openvslam', 'frame_trajectory_with_file_name.txt')
        pose_file_path = os.path.join(root_dir, sub_path)

        with open(pose_file_path) as f:
            lines = f.readlines()
                
            for line in lines:
                tokens = line.split()
                if tokens[0] == '#':
                    continue

                t, quat, img_fname = np.array(list(map(float, tokens[0:3]))), np.array(list(map(float, tokens[3:7]))), tokens[8]
                if img_ext:
                    img_fname = img_fname.split('.')[0] + img_ext

                rot = R.from_quat(quat).as_matrix()
                w2c = np.concatenate((rot, t[:,np.newaxis]), axis=1)
                w2c = np.concatenate((w2c, [[0,0,0,1]]), axis=0)
                c2w = np.linalg.inv(w2c)

                self.poses_dict[img_fname] = self.world_align @ c2w @ self.rays2cam


class Pix4dPoseDescriptor(PoseDescriptorBase):
    def __init__(self):
        super().__init__()
    
    @property
    def rays2cam(self):
        return np.array([
            [1., 0., 0., 0.],
            [0.,-1., 0., 0.],
            [0., 0.,-1., 0.],
            [0., 0., 0., 1.]
            ])

    @property
    def world_align(self):
        return np.eye(4)

    def read_pose_file(self, root_dir, sub_path=None, img_ext=None):
        if not sub_path:
            sub_path = os.path.join('pix4d', 'calibrated_camera_parameters.txt')
        pose_file_path = os.path.join(root_dir, sub_path)

        with open(pose_file_path) as f:
            lines = f.readlines()[3:] # discard file info
                
            for img_idx in range(len(lines) // 5):
                img_fname = lines[img_idx * 5 + 0].split()[0]
                if img_ext:
                    img_fname = img_fname.split('.')[0] + img_ext
                
                t = np.array(list(map(float, lines[img_idx * 5 + 1].split())))
                rot = np.array([
                    list(map(float, lines[img_idx * 5 + 2].split())),
                    list(map(float, lines[img_idx * 5 + 3].split())),
                    list(map(float, lines[img_idx * 5 + 4].split())),
                ])
                w2c = np.concatenate((rot, t[:,np.newaxis]), axis=1)
                w2c = np.concatenate((w2c, [[0,0,0,1]]), axis=0)
                c2w = np.linalg.inv(w2c)
                # c2w = w2c

                self.poses_dict[img_fname] = self.world_align @ c2w @ self.rays2cam


pose_descriptor_dict = {
    'colmap': ColmapPoseDescriptor,
    'openvslam': OpenVSlamPoseDescriptor,
    'pix4d': Pix4dPoseDescriptor,
}