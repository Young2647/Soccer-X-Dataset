import numpy as np
import blobfile as bf
import utils.common as common
from tqdm import tqdm
import utils.nn_transforms as nn_transforms
import itertools
import wandb
import torch
import os
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader
from torch_ema import ExponentialMovingAverage
from scipy.spatial.transform import Rotation as R

from diffusion.resample import create_named_schedule_sampler
from diffusion.gaussian_diffusion import *

from utils.config import const

class BaseTrainingPortal:
    def __init__(self, config, model, diffusion, dataloader, logger, tb_writer, prior_loader=None):
        
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.logger = logger
        self.tb_writer = tb_writer
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.lr_anneal_steps = config.trainer.lr_anneal_steps

        self.epoch = 0
        self.num_epochs = config.trainer.epoch
        self.save_freq = config.trainer.save_freq
        self.best_loss = 1e10
        
        print('Train with %d epoches, %d batches by %d batch_size' % (self.num_epochs, len(self.dataloader), self.batch_size))

        self.save_dir = config.save

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=config.trainer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=self.lr * 0.1)
        
        if config.trainer.ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        
        self.device = config.device

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.use_ddp = False
        
        self.prior_loader = prior_loader
        
        
    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        raise NotImplementedError('diffuse function must be implemented')

    def evaluate_sampling(self, dataloader, save_folder_name):
        raise NotImplementedError('evaluate_sampling function must be implemented')
    
        
    def run_loop(self):
        sampling_num = 16
        sampling_idx = np.random.randint(0, len(self.dataloader.dataset), sampling_num)
        sampling_subset = DataLoader(Subset(self.dataloader.dataset, sampling_idx), batch_size=sampling_num)
        self.evaluate_sampling(sampling_subset, save_folder_name='init_samples')
        
        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f'Epoch {self.epoch}')
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx
            epoch_losses = {}
            
            data_len = len(self.dataloader)
            
            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                x_start = datas['data']

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                
                _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                total_loss = (losses["loss"] * weights).mean()
                total_loss.backward()
                self.opt.step()
            
                if self.config.trainer.ema:
                    self.ema.update()
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())
            
            if self.prior_loader is not None:
                for prior_datas in itertools.islice(self.prior_loader, data_len):
                    prior_datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in prior_datas.items()}
                    prior_cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in prior_datas['conditions'].items()}
                    prior_x_start = prior_datas['data']
                    
                    self.opt.zero_grad()
                    t, weights = self.schedule_sampler.sample(prior_x_start.shape[0], self.device)
                    
                    _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                    total_loss = (prior_losses["loss"] * weights).mean()
                    total_loss.backward()
                    self.opt.step()
                    
                    for key_name in prior_losses.keys():
                        if 'loss' in key_name:
                            if key_name not in epoch_losses.keys():
                                epoch_losses[key_name] = []
                            epoch_losses[key_name].append(prior_losses[key_name].mean().item())
            
            loss_str = ''
            for key in epoch_losses.keys():
                loss_str += f'{key}: {np.mean(epoch_losses[key]):.6f}, '
            
            epoch_avg_loss = np.mean(epoch_losses['loss'])
            
            if self.epoch > 10 and epoch_avg_loss < self.best_loss:                
                self.save_checkpoint(filename='best')
            
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
            
            epoch_process_bar.set_description(f'Epoch {epoch_idx}/{self.config.trainer.epoch} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}')
            self.logger.info(f'Epoch {epoch_idx}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}')
                        
            if epoch_idx > 0 and epoch_idx % self.config.trainer.save_freq == 0:
                self.save_checkpoint(filename=f'weights_{epoch_idx}')
                self.evaluate_sampling(sampling_subset, save_folder_name='train_samples')
            
            wandb.log({'epoch': epoch_idx})
            for key_name in epoch_losses.keys():
                if 'loss' in key_name:
                    wandb.log({key_name: np.mean(epoch_losses[key_name])})

            self.scheduler.step()
        
        best_path = '%s/best.pt' % (self.config.save)
        self.load_checkpoint(best_path)
        self.evaluate_sampling(sampling_subset, save_folder_name='best')


    def state_dict(self):
        model_state = self.model.state_dict()
        opt_state = self.opt.state_dict()
            
        return {
            'epoch': self.epoch,
            'state_dict': model_state,
            'opt_state_dict': opt_state,
            'config': self.config,
            'loss': self.best_loss,
        }

    def save_checkpoint(self, filename='weights'):
        save_path = '%s/%s.pt' % (self.config.save, filename)
        with bf.BlobFile(bf.join(save_path), "wb") as f:
            torch.save(self.state_dict(), f)
        self.logger.info(f'Saved checkpoint: {save_path}')


    def load_checkpoint(self, resume_checkpoint, load_hyper=True):
        if bf.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            if load_hyper:
                self.epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint['loss']
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
            self.logger.info('\nLoad checkpoint from %s, start at epoch %d, loss: %.4f' % (resume_checkpoint, self.epoch, checkpoint['loss']))
        else:
            raise FileNotFoundError(f'No checkpoint found at {resume_checkpoint}')


class MotionTrainingPortal(BaseTrainingPortal):
    def __init__(self, config, model, diffusion, dataloader, logger, finetune_loader=None):
        super().__init__(config, model, diffusion, dataloader, logger, finetune_loader)
        
        # body model part
        self.body_bone = self.dataloader.dataset.T_pose.body_bone

    def diffuse(self, x_start, t, cond, noise=None, return_loss=False):
        batch_size, frame_num, joint_num, joint_feat = x_start.shape
        x_start = x_start.permute(0, 2, 3, 1)
        
        if noise is None:
            noise = th.randn_like(x_start)
        
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)
        
        # [bs, joint_num, joint_feat, future_frames]
        cond['past_motion'] = cond['past_motion'].permute(0, 2, 3, 1) # [bs, joint_num, joint_feat, past_frames]
        cond['traj_pose'] = cond['traj_pose'].permute(0, 2, 1) # [bs, 6, future_frame]
        cond['traj_trans'] = cond['traj_trans'].permute(0, 2, 1) # [bs, 2, future_frame]

        # out = self.diffusion.p_sample(self.model, x_t, t, model_kwargs={'cond': cond})
        # model_output = out['pred_xstart']
        model_output = self.model.interface(x_t, self.diffusion._scale_timesteps(t), cond)

        if return_loss:
            loss_terms = {}
            
            if self.diffusion.model_var_type in [ModelVarType.LEARNED,  ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                loss_terms["vb"] = self.diffusion._vb_terms_bpd(model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, clip_denoised=False)["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0
            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            mask = cond['mask'].view(batch_size, 1, 1, -1)
            
            if self.config.trainer.use_loss_mse:
                loss_terms['loss_data'] = 5 * self.diffusion.masked_l2(target, model_output, mask) # mean_flat(rot_mse)
                
            if self.config.trainer.use_loss_vel:
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                target_vel = target[..., 1:] - target[..., :-1]
                loss_terms['loss_data_vel'] = 10 * self.diffusion.masked_l2(target_vel[:, :-2], model_output_vel[:, :-2], mask[..., 1:])
                  
            if self.config.trainer.use_loss_3d or self.config.use_loss_contact:
                target_rot, pred_rot, past_rot = target.permute(0, 3, 1, 2), model_output.permute(0, 3, 1, 2), cond['past_motion'].permute(0, 3, 1, 2)
                target_root_pos, pred_root_pos, past_root_pos = target_rot[:, :, -5, :3], pred_rot[:, :, -5, :3], past_rot[:, :, -1, :3]  
            
                target_xyz = nn_transforms.smpl_FK(target_rot[:,:,:-5], self.body_bone, target_root_pos, rotation_type=self.config.arch.rot_req)
                pred_xyz = nn_transforms.smpl_FK(pred_rot[:,:,:-5], self.body_bone, pred_root_pos, rotation_type=self.config.arch.rot_req)
            
                if self.config.trainer.use_loss_3d:
                    loss_terms["loss_geo_xyz"] = 1 * self.diffusion.masked_l2(target_xyz.permute(0, 2, 3, 1), pred_xyz.permute(0, 2, 3, 1), mask)
                
                if self.config.trainer.use_loss_vel:
                    target_xyz_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
                    pred_xyz_vel = pred_xyz[:, 1:] - pred_xyz[:, :-1]
                    loss_terms["loss_geo_xyz_vel"] = 2 * self.diffusion.masked_l2(target_xyz_vel.permute(0, 2, 3, 1), pred_xyz_vel.permute(0, 2, 3, 1), mask[..., 1:])
                
                if self.config.trainer.use_loss_contact:
                    l_foot_idx, r_foot_idx = 10, 11
                    relevant_joints = [l_foot_idx, r_foot_idx]
                    target_xyz_reshape = target_xyz.permute(0, 2, 3, 1)  
                    pred_xyz_reshape = pred_xyz.permute(0, 2, 3, 1)
                    gt_joint_xyz = target_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 2, 3, Frames]
                    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
                    pred_joint_xyz = pred_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 2, 3, Frames]
                    pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                    pred_vel[~fc_mask] = 0
                    loss_terms["loss_foot_contact"] = self.diffusion.masked_l2(pred_vel,
                                                torch.zeros(pred_vel.shape, device=pred_vel.device),
                                                mask[:, :, :, 1:])
                
                if self.config.trainer.use_loss_ball_acc:
                    output_ball_contact = torch.max(model_output.permute(0, 3, 1, 2)[:,2:,-1:,4:], dim=-1)[0].unsqueeze(-1)
                    output_ball_vel = model_output.permute(0, 3, 1, 2)[:,1:,-3:-2,:3] - model_output.permute(0, 3, 1, 2)[:, :-1, -3:-2, :3]
                    output_ball_acc_horizon = output_ball_vel[:, 1:,:,[0,2]] - output_ball_vel[:, :-1, :,[0,2]]
                    output_ball_acc_horizon = torch.clamp(output_ball_acc_horizon, min=0.0) * const.FPS
                    pred_acc = ((1 - output_ball_contact) * output_ball_acc_horizon).permute(0, 2, 3, 1)
                    loss_terms["loss_ball_acc"] = 0.1* self.diffusion.masked_l2(pred_acc,
                                                                           torch.zeros(pred_acc.shape, device=pred_acc.device),
                                                                            mask[..., 2:])
                if self.config.trainer.use_loss_transition_vel:
                    target_transition_vel = target[...,:1] - cond['past_motion'][...,-1:]
                    pred_transition_vel = model_output[...,:1] - cond['past_motion'][...,-1:]
                    loss_terms["loss_transition_vel"] = self.diffusion.masked_l2(target_transition_vel, pred_transition_vel, mask[...,:1])

                if self.config.trainer.use_loss_ball_global:
                    gt_human_trans = target.permute(0, 3, 1, 2)[..., -5:-4, :3]
                    gt_ball_relative_pos = target.permute(0, 3, 1, 2)[...,-4:-3,:3]
                    gt_soccer_global_pos = gt_ball_relative_pos + gt_human_trans
                    gt_soccer_global_vel = gt_soccer_global_pos[:, 1:] - gt_soccer_global_pos[:, :-1]

                    pred_human_trans = model_output.permute(0, 3, 1, 2)[..., -5:-4, :3]
                    pred_ball_relative_pos = model_output.permute(0, 3, 1, 2)[..., -4:-3, :3]
                    pred_soccer_global_pos = pred_ball_relative_pos  + pred_human_trans
                    pred_soccer_global_vel = pred_soccer_global_pos[:, 1:] - pred_soccer_global_pos[:, :-1]

                    loss_terms["loss_ball_global"] = (self.diffusion.masked_l2(gt_soccer_global_pos.permute(0, 2, 3, 1), pred_soccer_global_pos.permute(0, 2, 3, 1), mask) + \
                                                     self.diffusion.masked_l2(gt_soccer_global_vel.permute(0, 2, 3, 1), pred_soccer_global_vel.permute(0, 2, 3, 1), mask[..., 1:]))

            loss_terms["loss"] = loss_terms.get('vb', 0.) + \
                            loss_terms.get('loss_data', 0.) + \
                            loss_terms.get('loss_data_vel', 0.) + \
                            loss_terms.get('loss_data_diff', 0.) + \
                            loss_terms.get('loss_geo_xyz', 0) + \
                            loss_terms.get('loss_geo_xyz_vel', 0) + \
                            loss_terms.get('loss_geo_xyz_diff', 0) + \
                            loss_terms.get('loss_foot_contact', 0) + \
                            loss_terms.get('loss_ball_acc', 0) + \
                            loss_terms.get('loss_transition_vel', 0) + \
                            loss_terms.get('loss_ball_global', 0) + \
                            loss_terms.get('loss_foot_contact_velocity', 0) 
            
            return model_output.permute(0, 3, 1, 2), loss_terms
        
        return model_output.permute(0, 3, 1, 2)
        
    
    def evaluate_sampling(self, dataloader, save_folder_name):
        self.model.eval()
        self.model.training = False
        common.mkdir('%s/%s' % (self.save_dir, save_folder_name))
        
        datas = next(iter(dataloader)) 
        datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
        cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
        x_start = datas['data']
        t, _ = self.schedule_sampler.sample(dataloader.batch_size, self.device)
        with torch.no_grad():
            model_output = self.diffuse(x_start, t, cond, noise=None, return_loss=False)
        
        common_past_motion = cond['past_motion'].permute(0, 3, 1, 2)
        self.export_samples(x_start, common_past_motion, '%s/%s/' % (self.save_dir, save_folder_name), 'gt')
        self.export_samples(model_output, common_past_motion, '%s/%s/' % (self.save_dir, save_folder_name), 'pred')
        
        self.logger.info(f'Evaluate the sampling {save_folder_name} at epoch {self.epoch}')
        

    def export_samples(self, future_motion_feature, past_motion_feature, save_path, prefix, cond=None):
        motion_feature = torch.cat((past_motion_feature, future_motion_feature), dim=1).detach()
        rotations = nn_transforms.repr6d2quat(motion_feature[:, :, :-4]).cpu().numpy()
        root_pos = motion_feature[:, :, -4, :3].cpu().numpy()
        relative_ball_pos = motion_feature[:, :, -3, :3].cpu().numpy()
        control_weight = motion_feature[:, :, -2, 0].cpu().numpy()
        contact = motion_feature[:, :, -1, :].cpu().numpy()

        for samplie_idx in range(future_motion_feature.shape[0]):
            T_pose_template = self.dataloader.dataset.T_pose.copy()
            T_pose_template.rotations = rotations[samplie_idx]
            T_pose_template.trans = root_pos[samplie_idx]
            root_rot_mat = R.from_quat(rotations[samplie_idx][:, 0, [1, 2, 3, 0]]).as_matrix()
            style = cond['style'][samplie_idx] if cond is not None else None

            if any(control_weight[samplie_idx]) > 0:
                temp_control_weight = np.where(control_weight[samplie_idx]==0, 0.01, control_weight[samplie_idx])
                ball_pos_temp = relative_ball_pos[samplie_idx] / temp_control_weight[:, np.newaxis] + root_pos[samplie_idx]
            else:
                ball_pos_temp = np.zeros((root_rot_mat.shape[0], 3))
                ball_pos_temp[:,2] += 10
            T_pose_template.ball_pos = ball_pos_temp
            
            T_pose_template.control_weight = control_weight[samplie_idx]
            T_pose_template.contact = contact[samplie_idx]

            filename = f'{save_path}/motion_{samplie_idx}.{prefix}'
            np.save(filename, {"smpl_pose": T_pose_template._rotations, "smpl_tran": T_pose_template._trans, 
                    "ball_pos": T_pose_template._ball_pos, "style": style})
        