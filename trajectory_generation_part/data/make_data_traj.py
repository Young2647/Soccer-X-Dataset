import sys
sys.path.append('.')
import os
import torch
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from pytorch3d import transforms
from utils.smpl_motion import Motion
from utils import smpl_motion_modules
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from utils.vis import traj_vis
from utils.config import style_set

def extract_traj(root_positions, forwards, smooth_kernel=[5, 10]):
    traj_trans, traj_angles, traj_poses = [], [], []
    FORWARD_AXIS = np.array([[0, 0, 1]]) # OpenGL system
    
    for kernel_size in smooth_kernel:
        smooth_traj = gaussian_filter1d(root_positions, kernel_size, axis=0, mode='nearest')
        smooth_traj = torch.from_numpy(smooth_traj)
        traj_trans.append(smooth_traj)
        
        forward = gaussian_filter1d(forwards, kernel_size, axis=0, mode='nearest')
        forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
        angle = np.arctan2(forward[:, 2], forward[:, 0])
        traj_angles.append(angle)
        
        v0s = FORWARD_AXIS.repeat(len(forward), axis=0)
        a = np.cross(v0s, forward)
        w = np.sqrt((v0s**2).sum(axis=-1) * (forward**2).sum(axis=-1)) + (v0s * forward).sum(axis=-1)
        between_wxyz = np.concatenate([w[...,np.newaxis], a], axis=-1)
        between = R.from_quat(between_wxyz[..., [1, 2, 3, 0]]).as_quat()[..., [3, 0, 1, 2]]
        between = torch.from_numpy(between)
        between = transforms.quaternion_to_axis_angle(between)
        traj_poses.append(between)
    
    return traj_trans, traj_angles, traj_poses

def process_motion(motion):
    motion, root_init_pos = smpl_motion_modules.root(motion, return_pos=True)
    motion = smpl_motion_modules.on_ground(motion)
    forward_angle, forwards = smpl_motion_modules.extract_forward(motion, np.arange(motion.frame_num),
                                                            16, 17, 1, 2, return_forward=True)
    traj_trans, traj_angles, traj_poses = extract_traj(motion.positions[:, 0], forwards, smooth_kernel=[5, 10])
    return traj_trans, traj_poses, root_init_pos

def make_traj_dataset(dataset_src_path, dataset_dst_path, window=55):
    data_list = []
    for clip_path in sorted(os.listdir(dataset_src_path), key=lambda x: int(x.replace("Part",""))):
        for file_name in tqdm(sorted(os.listdir(os.path.join(dataset_src_path, clip_path)))):
            data_path = os.path.join(dataset_src_path, clip_path, file_name)
            data = torch.load(data_path)
            if data['frames'] < window:
                continue
            if data['motion_label'] in ['fancy', 'stop']:
                continue
            try:
                frames = data['frames']
                style = data['motion_label']

                traj_trans = data['human_trans']
                traj_poses = data['human_poses'][:, :3]

                smpl_pose = data['human_poses'].reshape(-1, 24, 3)
                smpl_tran = data['human_trans']
                smpl_quat = transforms.axis_angle_to_quaternion(smpl_pose)
                smpl_motion = Motion(smpl_quat.numpy(), smpl_tran.numpy(), filepath=file_name)
                traj_tran_, traj_pose_, root_init_pos = process_motion(smpl_motion)
                traj_trans = traj_tran_[0]
                traj_poses = traj_pose_[0]
                # traj_vis(past_traj_trans=traj_trans, past_traj_poses=traj_poses, 
                #          opponent_pos=opponent_pos, traj_direction=traj_beat, title='traj_beat')
                # traj_vis(past_traj_trans=traj_trans, past_traj_poses=traj_poses, 
                #          opponent_pos=opponent_pos, traj_direction=traj_opponent, title='traj_opponent')
                
                data = {'traj_trans': traj_trans.float(),
                        'traj_poses': transforms.axis_angle_to_matrix(traj_poses).float(),
                        'style': style,
                        'file_name': data_path,
                }
                data_list.append(data)
            except :
                print(os.path.join(dataset_src_path, clip_path, file_name))

    pickle.dump(data_list, open(dataset_dst_path, 'wb'))
    print('Finish exporting %s' % dataset_dst_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='### traj data processing')
    parser.add_argument('-d', '--data_root', type=str, help='The root path of the dataset')
    parser.add_argument('-e', '--export_path', type=str, help='The export path of the processed data')
    parser.add_argument('-w', '--window', type=int, default=55, help='The window size of the motion')
    args = parser.parse_args()
    make_traj_dataset(args.data_root, args.export_path, args.window)