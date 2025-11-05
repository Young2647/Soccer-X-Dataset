import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pytorch3d import transforms
from utils import nn_transforms
import socket
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_6d_to_axis_angle(r6d):
    matrix = transforms.rotation_6d_to_matrix(r6d)
    axis = transforms.matrix_to_axis_angle(matrix)
    return axis

def axis_angle_to_rotation_6d(axis):
    matrix = transforms.axis_angle_to_matrix(axis)
    r6d = transforms.matrix_to_rotation_6d(matrix)
    return r6d

def traj_vis(past_traj_trans=None, past_traj_poses=None, 
             future_traj_trans=None, future_traj_poses=None, 
             opponent_pos=None, traj_direction=None, title='', 
             save_path=None, show=True):
    r'''
    input: 
        trans: [N, 2] or [N, 3]
        poses: [N, 6] or [N, 3]
        opponent_pos: [2]
        traj_direction: [2]
    '''
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # draw past traj
    if past_traj_trans != None:
        past_traj_trans = past_traj_trans.squeeze()
        if past_traj_trans.shape[1] == 3:
            past_traj_trans = past_traj_trans[:, [0, 2]]
        ax.scatter(past_traj_trans[:, 0], past_traj_trans[:, 1], label=f'past traj', color='orange')

        if past_traj_poses != None:
            past_traj_poses = past_traj_poses.squeeze()
            if past_traj_poses.shape[1] == 6:
                past_traj_poses = rotation_6d_to_axis_angle(past_traj_poses)
            if len(past_traj_poses.shape) == 3:
                past_traj_poses = transforms.matrix_to_axis_angle(past_traj_poses)
            for i in range(past_traj_poses.shape[0]):
                axis_angle = past_traj_poses[i]
                rotation = R.from_rotvec(axis_angle)
                direction = rotation.apply([0, 0, 1]) / 10
                ax.arrow(past_traj_trans[i, 0], past_traj_trans[i, 1], direction[0], direction[2],
                            head_width=0.01, head_length=0.01, fc='black', ec='black')

    # draw future traj
    if future_traj_trans != None:
        future_traj_trans = future_traj_trans.squeeze()
        if future_traj_trans.shape[1] == 3:
            future_traj_trans = future_traj_trans[:, [0, 2]]
        ax.scatter(future_traj_trans[:, 0], future_traj_trans[:, 1], label=f'future traj', color='blue')

        if future_traj_poses != None:
            future_traj_poses = future_traj_poses.squeeze()
            if future_traj_poses.shape[1] == 6:
                future_traj_poses = rotation_6d_to_axis_angle(future_traj_poses)
            if len(future_traj_poses.shape) == 3:
                future_traj_poses = transforms.matrix_to_axis_angle(future_traj_poses)
            for i in range(future_traj_poses.shape[0]):
                axis_angle = future_traj_poses[i]
                rotation = R.from_rotvec(axis_angle)
                direction = rotation.apply([0, 0, 1]) / 10
                ax.arrow(future_traj_trans[i, 0], future_traj_trans[i, 1], direction[0], direction[2],
                            head_width=0.01, head_length=0.01, fc='black', ec='black')

    if opponent_pos != None:
        opponent_pos = opponent_pos.squeeze()
        if opponent_pos.shape[0] == 3:
            opponent_pos = opponent_pos[[0, 2]]
        ax.scatter(opponent_pos[0], opponent_pos[1], label=f'opponent pos', color='red')

    if traj_direction != None:
        traj_direction = traj_direction.squeeze()
        if traj_direction.shape[0] == 3:
            traj_direction = traj_direction[[0, 2]]
        traj_direction = traj_direction / torch.norm(traj_direction)
        ax.arrow(0, 0, traj_direction[0], traj_direction[1],
                        head_width=0.01, head_length=0.01, fc='black', ec='black')
        
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(title)
    # plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    
    plt.close(fig)
    
        

def cope_tensor(message):
    return ','.join(['%g' % v for v in message])  + '#'
def cope_string(message):
    return message + '#'

def package_message(data):
    s = "@"
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            s = s + key + '&'
            s += cope_tensor(value)
        else:
            s = s + key + '&'
            s += cope_string(value)
    return s

def soccer_vis(data, file_name='debug', fps=30):
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    conn, addr = server_for_unity.accept()
    from pygame.time import Clock
    clock = Clock()

    human_poses = data['human_poses']                       # [N, 72] 
    human_trans = data['human_trans']                       # [N, 3]
    soccer_pos = data['soccer_pos']                         # [N, 3]
    soccer_ori = data['soccer_ori']                         # [N, 3]
    foot_ground_contact = data['foot_ground_contact']       # [N, 4]
    foot_ball_contact = data['foot_ball_contact']           # [N, 2]
    foot_pos_ = data['foot_pos']                             # [N, 4, 3]
    soccer_ori = transforms.axis_angle_to_quaternion(soccer_ori)

    while(1):
        for frame, (human_p, human_t, soccer_t, soccer_o, contact_ground, contact_ball, foot_pos) in enumerate(zip(human_poses, human_trans, soccer_pos, soccer_ori, foot_ground_contact, foot_ball_contact, foot_pos_)):
            clock.tick(fps)
            data = {
                "human_p": human_p,
                "human_t": human_t,
                "soccer_t": soccer_t,
                "soccer_o": soccer_o,
                "frame": str(frame).zfill(6),
                "contact_ground": contact_ground,
                "contact_ball": contact_ball,
                "foot_pos_0": foot_pos[0],
                "foot_pos_1": foot_pos[1],
                "foot_pos_2": foot_pos[2],
                "foot_pos_3": foot_pos[3],
                "file_name": str(file_name)
            }

            s = package_message(data)
            conn.send(s.encode('utf8'))

def vis_smpl(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    rotations = nn_transforms.repr6d2quat(data[:, :, :-1])
    root_pos = data[:, :, -1, :3]
    
    poses = transforms.quaternion_to_axis_angle(rotations).reshape(-1, 72)
    trans = root_pos.reshape(-1, 3)
    
    frames = trans.shape[0]
    data = {'human_poses': poses,
            'human_trans': trans,
            'soccer_pos': torch.zeros(frames, 3),
            'soccer_ori': torch.zeros(frames, 3),
            'foot_ground_contact': torch.zeros(frames, 4),
            'foot_ball_contact': torch.zeros(frames, 2),
            'foot_pos': torch.zeros(frames, 4, 3),
            }
    soccer_vis(data)