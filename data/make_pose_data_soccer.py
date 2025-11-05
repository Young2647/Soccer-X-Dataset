import sys
sys.path.append('./')

import os
import numpy as np
import torch
from utils.smpl_motion import Motion
from pytorch3d import transforms
from utils import smpl_motion_modules
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from utils.config import const, style_set, ignore_list
from tqdm import tqdm
import pickle
import argparse


def get_fooot_ground_contact(foot_pos):
    ankle_height = foot_pos[:, :2, 1]      # idx = 7, 8
    foot_height = foot_pos[:, -2:, 1]      # idx = 10, 11
    height_judge = torch.concat(((ankle_height < 0.1), (foot_height < 0.05)), dim=-1)

    foot_vel = torch.zeros(foot_pos.shape[:2])
    foot_vel[:-1] = (foot_pos[1:] - foot_pos[:-1]).norm(dim=-1)
    vel_judge = foot_vel < 0.01       

    fooot_ground_contact = (height_judge & vel_judge).int()
    return fooot_ground_contact

def get_foot_ball_contact(foot_pos, soccer_pos):
    foot_pos = torch.concat((foot_pos, 
                                ((foot_pos[:, 0] + foot_pos[:, 2])/2).unsqueeze(1), 
                                ((foot_pos[:, 1] + foot_pos[:, 3])/2).unsqueeze(1)), 
                                dim=1)
    foot_distance = (foot_pos - soccer_pos.unsqueeze(1)).norm(dim=-1)
    ankle_contact = (foot_distance[:, [0,1]] < 0.15) | ((foot_pos[:, [0,1], 1] > 0.1)  & (foot_distance[:, [0,1]] < 0.3))
    foot_contact =  (foot_distance[:, [2,3]] < 0.15) | ((foot_pos[:, [2,3], 1] > 0.05) & (foot_distance[:, [2,3]] < 0.3))
    mid_contact  =  (foot_distance[:, [4,5]] < 0.15) | ((foot_pos[:, [4,5], 1] > 0.05) & (foot_distance[:, [4,5]] < 0.3))
    foot_ball_contact = (ankle_contact | foot_contact | mid_contact).int()
    return foot_ball_contact

def extract_traj(root_positions, forwards, smooth_kernel=[5, 10]):
    traj_trans, traj_angles, traj_poses = [], [], []
    FORWARD_AXIS = np.array([[0, 0, 1]]) # OpenGL system
    
    for kernel_size in smooth_kernel:
        smooth_traj = gaussian_filter1d(root_positions[:, [0, 2]], kernel_size, axis=0, mode='nearest')
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
        
        traj_poses.append(between)
    
    return traj_trans, traj_angles, traj_poses

def process_motion(motion):
    motion, root_init_pos = smpl_motion_modules.root(motion, return_pos=True)
    motion = smpl_motion_modules.on_ground(motion)
    _, forwards = smpl_motion_modules.extract_forward(motion, np.arange(motion.frame_num),
                                                            16, 17,
                                                            1, 2,return_forward=True)
    traj_trans, traj_angles, traj_poses = extract_traj(motion.positions[:, 0], forwards, smooth_kernel=[5, 10])
    return motion, {
        'local_joint_rotations': motion.rotations,
        'global_root_positions': motion.positions[:, 0],
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses,
        'root_init_pos': root_init_pos
    }

def process_motion_traj_only(motion):
    _, forwards = smpl_motion_modules.extract_forward(motion, np.arange(motion.frame_num),
                                                            16, 17,
                                                            1, 2,return_forward=True)
    traj_trans, traj_angles, traj_poses = extract_traj(motion.positions[:, 0], forwards, smooth_kernel=[5, 10])
    return motion, {
        'local_joint_rotations': motion.rotations,
        'global_root_positions': motion.positions[:, 0],
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }

def process_soccer_cond(data, motion_dict):
    ball_pos = data['soccer_pos']
    ball_pos[:, [0,2]] -= motion_dict['root_init_pos']
    ball_rot = transforms.axis_angle_to_quaternion(data['soccer_ori'])
    human_pos = data['human_trans']
    control_weight = 1.0 - torch.norm((ball_pos[:, [0,2]] - human_pos[:,[0,2]]), dim=-1)/const.CONTROL_RADIUS
    control_weight = torch.clamp(control_weight, 0, 1)
    motion_dict['control_weight'] = control_weight.numpy()
    motion_dict['ball_pos'] = ball_pos.view(-1,3).numpy()
    motion_dict['ball_rot'] = ball_rot.view(-1,4).numpy()
    foot_pos = torch.from_numpy(smpl_motion.positions[:, [7,8,10,11]])   # idx = 7,8,10,11
    foot_ground_contact = get_fooot_ground_contact(foot_pos)
    foot_ball_contact = get_foot_ball_contact(foot_pos, ball_pos)
    motion_dict['contact'] = torch.cat([foot_ground_contact, foot_ball_contact], dim=-1).numpy()
    return motion_dict
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='### pose data processing')
    parser.add_argument('-d', '--data_root', type=str, nargs='+', help='The root path of the dataset')
    parser.add_argument('-e', '--export_path', type=str, help='The export path of the processed data')
    parser.add_argument('-w', '--window', type=int, default=55, help='The window size of the motion')
    args = parser.parse_args()

    data_root = args.data_root
    export_path = args.export_path
    window = args.window

    data_list = {
        'names':None,
        'motions': []
    }

    for root_path in data_root:
        print('Processing %s' % root_path)
        for clip_path in sorted(os.listdir(root_path), key=lambda x: int(x.replace("Part",""))):
            print('Processing %s' % clip_path)
            for pt_path in tqdm(sorted(os.listdir(os.path.join(root_path, clip_path)))):
                _data = torch.load(os.path.join(root_path, clip_path, pt_path))
                data_shape = _data['human_poses'].shape[0]
                # if data_shape <= window:
                #     print('Too short! Skip %s' % pt_path)
                if data_shape <= 20:
                    print('Too short! Skip %s' % pt_path)
                    continue
                if _data['soccer_pos'].shape[0] != data_shape:
                    print('Inconsistent shape! Skip %s' % pt_path)
                    continue
                if _data['motion_label'] not in style_set.demo_style_set:
                    print('Not in style set! Skip %s' % pt_path)
                    continue
                if pt_path in ignore_list.ignore_list:
                    print('In ignore list! Skip %s' % pt_path)
                    continue
                smpl_pose = _data['human_poses'].reshape(-1, 24, 3)
                smpl_tran = _data['human_trans']
                smpl_quat = transforms.axis_angle_to_quaternion(smpl_pose)
                smpl_motion = Motion(smpl_quat.numpy(), smpl_tran.numpy(), filepath=pt_path)
                smpl_motion, motion_data = process_motion(smpl_motion)
                motion_data['style'] = _data['motion_label']
                motion_data = process_soccer_cond(_data, motion_data)
                data_list["motions"].append(motion_data)
                print('Finish processing %s' % pt_path)

    T_rotation = np.zeros((1, smpl_motion.rotations.shape[1], smpl_motion.rotations.shape[2]))
    T_rotation[..., 0] = 1
    T_position = np.zeros((1, smpl_motion.positions.shape[1], smpl_motion.positions.shape[2]))
    T_trans = np.zeros((1, smpl_motion.trans.shape[1]))
    T_ball_pos = np.zeros((1, 3))
    T_ball_vel = np.zeros((1, 3))
    T_control_weight = np.zeros((1, 1))
    T_contact = np.zeros((1, 6))

    smpl_motion.rotations = T_rotation
    smpl_motion.positions = T_position
    smpl_motion.trans = T_trans
    smpl_motion.ball_pos = T_ball_pos
    smpl_motion.ball_vel = T_ball_vel
    smpl_motion.control_weight = T_control_weight
    smpl_motion.contact = T_contact
    
    data_list['T_pose'] = smpl_motion    

    pickle.dump(data_list, open(export_path, 'wb'))
    print('Finish exporting %s' % export_path)
    