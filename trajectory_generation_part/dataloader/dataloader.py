import sys
sys.path.append('./')

import os
import random
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pytorch3d import transforms

import utils.nn_transforms as nn_transforms
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader
from utils.vis import traj_vis
from utils.config import style_set


class Soccer_Dataset(Dataset):
    def __init__(self, args, model='test', offset_frame=1, past_frame=10, future_frame=45):
        self.model = model
        self.pkl_path = args.dataset_path
        self.window_size = past_frame + future_frame
        self.future_frame = future_frame
        self.past_frame = past_frame
        self.reference_frame_idx = past_frame - 1
        self.data_list = {'traj_trans': [], 
                          'traj_poses': []}
        self.cond_list = {'style': [],
                          'traj_direction': []}
        self.debug_list = {'file_name': []}
        data_source = pickle.load(open(self.pkl_path, 'rb'))
        frame_nums = []
        item_frame_indices_list = [] # store the frame index for each clip. shape = 1 + window_size, 1 represent the motion idx
        motion_idx = 0
        for motion_item in tqdm(data_source):
            frame_num = motion_item['traj_trans'].shape[0]
            if frame_num < self.window_size:
                continue

            frame_nums.append([motion_item['style'], frame_num])
            self.data_list['traj_trans'].append(motion_item['traj_trans'].float())
            self.data_list['traj_poses'].append(motion_item['traj_poses'].float())
            self.cond_list['style'].append(motion_item['style'])
            self.debug_list['file_name'].append(motion_item['file_name'])

            clip_indices = np.arange(0, frame_num - self.window_size + 1, offset_frame)[:, None] + np.arange(self.window_size)
            clip_indices_with_idx = np.hstack((np.full((len(clip_indices), 1), motion_idx, dtype=clip_indices.dtype), clip_indices))            
            item_frame_indices_list.append(clip_indices_with_idx)
            motion_idx += 1
            
        self.item_frame_indices = np.concatenate(item_frame_indices_list, axis=0)
        # self.style_set = sorted(list(set(self.cond_list['style'])))
        self.style_set = style_set.demo_style_set
        print('Dataset loaded, trained with %d clips, %d frames, %d mins in total' % (len(frame_nums), sum([item[1] for item in frame_nums]), sum([item[1] for item in frame_nums])/30/60))
        
    def __len__(self):
        return len(self.item_frame_indices)

    def __getitem__(self, idx):
        # get data
        item_frame_indice = self.item_frame_indices[idx]
        motion_idx, frame_indices = item_frame_indice[0], item_frame_indice[1:]
        
        traj_trans = self.data_list['traj_trans'][motion_idx][frame_indices].clone().detach()
        traj_poses = self.data_list['traj_poses'][motion_idx][frame_indices].clone().detach()
        style = self.cond_list['style'][motion_idx]
        style_idx = float(self.style_set.index(style))
        # file_name = self.debug_list['file_name'][motion_idx]

        # process traj direction ###############################################################
        # traj_beat = traj_direction['traj_beat'].clone().detach()
        # traj_opponent = traj_direction['traj_opponent'].clone().detach()

        # traj_direction = traj_beat
        ########################################################################################

        # vis pkl ############################################################################
        # traj_vis(past_traj_trans=traj_trans, past_traj_poses=traj_poses, 
        #          opponent_pos=opponent_pos, traj_direction=traj_direction, title='pkl')
        ######################################################################################
        
        # align ##############################################################################
        reference_trans = traj_trans[self.reference_frame_idx].clone()
        traj_trans -= reference_trans

        # reference_poses = traj_poses[self.reference_frame_idx].clone()
        # traj_trans = torch.matmul(reference_poses.T, traj_trans.unsqueeze(-1)).squeeze(-1)
        # traj_poses = torch.matmul(traj_poses, reference_poses.T)
        # opponent_pos = torch.matmul(reference_poses.T, opponent_pos.unsqueeze(-1)).squeeze(-1)
        # traj_direction = torch.matmul(reference_poses.T, traj_direction.unsqueeze(-1)).squeeze(-1)
        ######################################################################################

        # vis aligned ########################################################################
        # traj_vis(past_traj_trans=traj_trans, past_traj_poses=traj_poses, 
        #          opponent_pos=opponent_pos, traj_direction=traj_direction, title='aligned')
        ######################################################################################

        # random rotation ####################################################################
        def random_rotation_matrix_around_y():
            theta = torch.rand(1) * 2 * torch.pi
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrix = torch.tensor([
                [cos_theta, torch.zeros(1), sin_theta],
                [torch.zeros(1), torch.ones(1), torch.zeros(1)],
                [-sin_theta, torch.zeros(1), cos_theta]
            ]).squeeze()
            return rotation_matrix

        if self.model == 'train':
            rotation_matrix = random_rotation_matrix_around_y()
            traj_trans = torch.matmul(rotation_matrix, traj_trans.unsqueeze(-1)).squeeze(-1)
            traj_poses = torch.matmul(rotation_matrix, traj_poses)
            # traj_direction = torch.matmul(rotation_matrix, traj_direction.unsqueeze(-1)).squeeze(-1)
        ######################################################################################

        # vis rotation #######################################################################
        # traj_vis(past_traj_trans=traj_trans, past_traj_poses=traj_poses, 
        #          opponent_pos=opponent_pos, traj_direction=traj_direction, title='rotation')
        ######################################################################################


        traj_trans = traj_trans[:, [0, 2]]
        traj_poses = transforms.matrix_to_rotation_6d(traj_poses)
        traj_trans_past = traj_trans[:self.past_frame]
        traj_poses_past = traj_poses[:self.past_frame]
        traj_trans_future = traj_trans[-self.future_frame:]
        traj_poses_future = traj_poses[-self.future_frame:]
        
        # traj_direction = traj_direction[[0, 2]]
        # traj_direction = traj_direction / torch.norm(traj_direction)
        

        data = {'data': {'traj_trans_future': traj_trans_future,
                         'traj_poses_future': traj_poses_future},
                'cond': {'traj_trans_past': traj_trans_past,
                         'traj_poses_past': traj_poses_past,
                         'dest_trans': traj_trans_future[-1],
                         'dest_poses': traj_poses_future[-1],
                         'style': style,
                         'style_idx': style_idx}}
        return data
        
if __name__ == '__main__':
    from config.config import get_args
    args = get_args()
    dataset = Soccer_Dataset(args, model='train')
    dataset[0]