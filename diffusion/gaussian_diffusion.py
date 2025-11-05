# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood

from utils import nn_transforms
from utils.config import const

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "linear1":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.01
        beta_end = scale * 0.7
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "linear2":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas
        beta_start = scale * 0.01
        beta_end = scale * 0.7
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_3d=1.,
        lambda_vel=1.,
        lambda_r_vel=1.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_3d = lambda_3d
        self.lambda_vel = lambda_vel
        self.lambda_r_vel = lambda_r_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        if self.lambda_3d > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.
        self.body_bone = torch.load(r'utils/body_bone.pt')

    def l2(self, a, b):
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss)
        n_entries = a.shape[1] * a.shape[2]
        mse_loss_val = loss / n_entries
        return mse_loss_val

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model.model.interface(x, self._scale_timesteps(t), **model_kwargs)
        # contact = model_output[:, -1, -2:, :].permute(0, 2, 1)

        # if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
        #     inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
        #     assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        #     assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
        #     model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
        #     # print('model_output', model_output.shape, model_output)
        #     # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
        #     # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        #################################################################################
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        #################################################################################
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def gradients_0724(self, x, cond):
        # x = torch.cat((cond['past_motion'], x), dim=-1)
        bs, _, _, frames = x.shape
        
        with torch.enable_grad():
            x.requires_grad_(True)
            x_ = x.permute(0, 3, 1, 2).contiguous()     # [BS, T, J, D]
            # contact
            human_poses_pred = x_[:, :, :24]
            human_trans_pred = x_[:, :, -4, :3]
            relative_soccer_pos = x_[:, :, -3, :3]
            control_weight = x_[:, :, -2, 0]
            temp_control_weight = torch.where(control_weight == 0, torch.tensor(1e-8, dtype=control_weight.dtype).to(x_.device), control_weight).unsqueeze(-1)
            soccer_pos = relative_soccer_pos / temp_control_weight + human_trans_pred

            def get_foot_ball_contact(foot_pos, soccer_pos):
                foot_pos_ = torch.concat((foot_pos, 
                                            ((foot_pos[:, :, 0] + foot_pos[:, :, 2])/2).unsqueeze(-2), 
                                            ((foot_pos[:, :, 1] + foot_pos[:, :, 3])/2).unsqueeze(-2)), 
                                            dim=-2)
                foot_distance = (foot_pos_ - soccer_pos.unsqueeze(-2)).norm(dim=-1)
                ankle_contact = (foot_distance[:, :, [0,1]] < 0.15) | ((foot_pos_[:, :, [0,1], 1] > 0.1)  & (foot_distance[:, :, [0,1]] < 0.3))
                foot_contact =  (foot_distance[:, :, [2,3]] < 0.15) | ((foot_pos_[:, :, [2,3], 1] > 0.05) & (foot_distance[:, :, [2,3]] < 0.3))
                mid_contact  =  (foot_distance[:, :, [4,5]] < 0.15) | ((foot_pos_[:, :, [4,5], 1] > 0.05) & (foot_distance[:, :, [4,5]] < 0.3))
                foot_ball_contact = (ankle_contact | foot_contact | mid_contact).int()
                return foot_ball_contact
            
            def get_foot_ground_contact(foot_pos):
                ankle_height = foot_pos[:, :, :2, 1]      # idx = 7, 8
                foot_height = foot_pos[:, :, -2:, 1]      # idx = 10, 11
                height_judge = torch.concat(((ankle_height < 0.1), (foot_height < 0.05)), dim=-1)

                foot_vel = torch.zeros(foot_pos.shape[:-1]).to(foot_pos.device)
                foot_vel[:, :-1] = (foot_pos[:, 1:] - foot_pos[:, :-1]).norm(dim=-1)
                vel_judge = foot_vel < 0.01       

                fooot_ground_contact = (height_judge & vel_judge).int()
                return fooot_ground_contact

            def get_angles_from_trans(trans, window_past=5, window_future=5, fps=30):
                r'''
                input: 
                trans: [BS, T, 2]
                output:
                angles: [BS, T, 1]
                '''
                bs, t, _ = trans.shape
                # get velocity
                vel_past = torch.zeros_like(trans)
                vel_future = torch.zeros_like(trans)
                for i in range(1, t-1):
                    start_idx = max(i-window_past, 0)
                    end_idx = min(i+window_future, t-1)
                    start_frames = i - start_idx
                    end_frames = end_idx - i
                    vel_past[:, i] = (trans[:, i] - trans[:, start_idx]) / start_frames * fps
                    vel_future[:, i] = (trans[:, end_idx] - trans[:, i]) / end_frames * fps
                vel_past[:, 0], vel_past[:, -1] = vel_past[:, 1], vel_past[:, -2]
                vel_future[:, 0], vel_future[:, -1] = vel_future[:, 1], vel_future[:, -2]
                angles_mask = torch.where((torch.norm(vel_past, dim=-1, keepdim=True) > 0.15) & (torch.norm(vel_future, dim=-1, keepdim=True) > 0.15), 1, 0)
                vel_past = vel_past / (torch.norm(vel_past, dim=-1, keepdim=True) + 1e-8)
                vel_future = vel_future / (torch.norm(vel_future, dim=-1, keepdim=True) + 1e-8)
                # get angle
                dot_product = torch.sum(vel_past * vel_future, dim=-1, keepdim=True)
                angles = torch.acos(dot_product.clamp(-1.0, 1.0)) * angles_mask
                return angles
            
            # calculate values
            joints_pos = nn_transforms.smpl_FK(human_poses_pred, self.body_bone, human_trans_pred, rotation_type='6d')
            # joints_pos = torch.zeros(bs, frames, 24, 3).to(x.device)      
            foot_pos = joints_pos[:, :, [7,8,10,11]]                # [BS, T, 4, 3]
            foot_ball_contact_pred = get_foot_ball_contact(foot_pos, soccer_pos)                                                    # [BS, T, 2]
            foot_ball_contact_pred = foot_ball_contact_pred.sum(dim=-1, keepdim=True).clamp(0, 1)                                   # [BS, T, 1]   
            foot_ground_contact_pred = get_foot_ground_contact(foot_pos)                                   # [BS, T, 1]    
            foot_ball_distance = torch.norm(foot_pos - soccer_pos.unsqueeze(-2), dim=-1)    # [BS, T, 6]
            
            ANGLES_THRESHOLD = 30 / 180 * torch.pi
            soccer_trans_2d = soccer_pos[:, :, [0, 2]]              # [BS, T, 2]
            soccer_angles = get_angles_from_trans(soccer_trans_2d, window_past=2, window_future=2)  # [BS, T, 1]
            contact_from_soccer_traj = (soccer_angles > ANGLES_THRESHOLD).int()

            r'''
            cost function:
            1. if soccer trajectory angle > ANGLES_THRESHOLD, foot-soccer-contact should be 1
           '''
            # 根据foot_ground_contact_pred判断哪个关节与球接触，并计算接触距离
            foot_ground_not_contact_pred = torch.where(foot_ground_contact_pred == 0, 1, 2)                                       # [BS, T, 4]
            foot_ball_distance_ = torch.min(foot_ground_not_contact_pred * foot_ball_distance, dim=-1, keepdim=True).values       # [BS, T, 1]
            # 根据cost function计算损失
            contact_flag = torch.where((contact_from_soccer_traj == 1) & (foot_ball_contact_pred == 0), 1, 0)
            contact_frames = contact_flag.sum(dim=[-1, -2])
            contact_distance = contact_flag * (foot_ball_distance_ - const.BALL_RADIUS)
            loss_soccer_contact = torch.mean(contact_distance.sum(dim=[-1, -2]) / (contact_frames + 1e-20))

            # output_ball_contact = torch.max(x_[:,1:,-1:,4:], dim=-1)[0].unsqueeze(-1)
            # output_ball_vel = soccer_pos[:, 1:] - soccer_pos[:, :-1]
            # output_ball_acc_horizon = output_ball_vel[:, 1:,[0,2]] - output_ball_vel[:, :-1, [0,2]]
            # output_ball_acc_horizon = torch.clamp(output_ball_acc_horizon, min=0.0)
            # pred_acc = ((1 - output_ball_contact) * output_ball_acc_horizon)
            # loss_acc = 10 * torch.mean(pred_acc ** 2) 

            loss = loss_soccer_contact
            print('loss:', loss)
            grad = torch.autograd.grad([loss.sum()], [x])[0]
            x.detach()
        return loss, grad


    def calc_grad_scale(self, mask_hint):
        # assert mask_hint.shape[1] == 196
        # print(mask_hint.shape)
        num_keyframes = mask_hint.sum(dim=-1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        # print(num_keyframes.shape)
        # print(max_keyframes)
        scale = 20 / max_keyframes
        # print('scale:',scale)
        return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def calc_grad_scale_kp(self, mask_hint):
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def guide_0724(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):

        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance
        
        # n_guide_steps = 10
        obj_verts = model_kwargs['y']['obj_points']
        obj_normals = model_kwargs['y']['obj_normals']
        # print(obj_verts.shape)

        n_guide_steps = 1
        # if train:
        #     if t[0] < 20:
        #         n_guide_steps = 100
        #     else:
        #         n_guide_steps = 20
        # else:
        #     if t[0] < 10:
        #         n_guide_steps = 500
        #     else:
        #         n_guide_steps = 10

        scale = 20
        for _ in range(n_guide_steps):
            loss, grad = self.gradients_0724(x, obj_verts, obj_normals)
            grad = model_variance * grad
            # print(grad)
            # print(loss.sum())
            # if t[0] >= t_stopgrad:
                # x = x - scale * grad
            # print(grad)
            x = x - scale * grad
        # print(loss.shape)
        log_guide = torch.mean(loss)
        index = t[0].cpu().item()
        self.log_guide_dict[index] = log_guide.cpu().item()
        print(f"{t[0]} step Guide: ",log_guide)
        return x

    def guide_0724_DSG(self, out, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        BS, J, D, T = x.shape
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance
        
        # n_guide_steps = 10

        ## DSG parts
        loss, grad = self.gradients_0724(out['pred_xstart'], cond=model_kwargs['y'])
        noise = th.randn_like(x)
        nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        r = torch.linalg.norm(sample.contiguous().view(BS, J, D*T) - out["mean"].contiguous().view(BS, J, D*T), dim=(1,2)).mean()
        if t[0] % 1 == 0:
            eps = 1e-20
            guidance_rate = 0.5
            grad_norm = torch.linalg.norm(grad.contiguous().view(BS, J, D*T), dim=(1,2))
            grad_norm = grad_norm.reshape(grad_norm.shape[0], 1, 1, 1)
            # print(f'grad_norm:{grad_norm.shape}')
            grad2 = grad / (grad_norm + eps)
            direction1 = -r * grad2
            direction2 = sample - out["mean"]
            mix_direction = direction2 + guidance_rate * (direction1 - direction2)
            mix_direction_norm = torch.linalg.norm(mix_direction.contiguous().view(BS, J, D*T), dim=(1,2))
            mix_direction_norm = mix_direction_norm.reshape(mix_direction_norm.shape[0], 1, 1, 1)
            # print(f'mix_direction_norm:{mix_direction_norm.shape}')
            mix_step = mix_direction / (mix_direction_norm + eps) * r
            x = out["mean"] + mix_step
        else:
            x = sample

        # log_guide = torch.mean(loss)
        # index = t[0].cpu().item()
        # self.log_guide_dict[index] = log_guide.cpu().item()
        # print(f"{t[0]} step Guide: ",log_guide)
        return x


    def guide(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # print(t_stopgrad)
        # print(scale)
        # print(min_variance)
        # print(model_kwargs.keys())
        # print(model_kwargs['y'].keys())
        # n_joint = 22 if x.shape[1] == 263 else 21
        n_joint = 17
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        length = model_kwargs['y']['lengths'].clone().detach().long()
        # print(length.shape)
        # print(type(length))
        # print(type(length[0]))
        # length = model_kwargs['y']['length'].clone().detach()
        # print(hint.shape)
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0
        # print(mask_hint.shape)
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        hint = hint * self.global_std + self.global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        # joint id
        # joint_ids = []
        # for m in mask_hint:
        #     joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
        #     joint_ids.append(joint_id)
        
         ##############################################
        if not train:
            scale = self.calc_grad_scale(mask_hint)
        ##################################################
        # if t[0] < 150:
        #     scale = 5
        # else:
        #     scale = 20

        # scale = 1
        # scale = 20
        # if t[0] < 50:
        #     scale = 2
        # else:
        #     scale = 20
        mask = model_kwargs['y']['mask']
        for _ in range(n_guide_steps):
            loss, grad = self.gradients(x, hint, mask_hint,mask,length)
            # print(grad.shape)
            # print(grad[0,:,0,1])
            # print(grad[0,:,0,0])
            grad = model_variance * grad
            # print(grad)
            # print(loss.sum())
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print(loss.shape)
        log_guide = torch.mean(loss)
        index = t[0].cpu().item()
        self.log_guide_dict[index] = log_guide.cpu().item()
        print(f"{t[0]} step Guide: ",log_guide)
        # print(f"{t[0]} step Guide: ",self.log_guide_dict)
        # with open('/root/code/OmniControl/save/0125_with_global_loss_no_guide/samples_0125_with_global_loss_no_guide_000070000_seed10_predefined/log_guide.pkl','wb')as f:
        #         pickle.dump(self.log_guide_dict,f)
        # # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_new(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        n_joints = 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        # bs,nf,_,_ = 
        # print(hint.shape)
        mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(dim=-1, keepdim=True) != 0
        # print(mask_hint.shape)
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        hint = hint.view(hint.shape[0], hint.shape[1],n_joints,3) * mask_hint
        # joint id
        # joint_ids = []
        # for m in mask_hint:
        #     joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
        #     joint_ids.append(joint_id)
        
        ##############################################
        # if not train:
        #     scale = self.calc_grad_scale_kp(mask_hint)
        ##################################################
        if t[0] < 150:
            scale = 5
        else:
            scale = 10

        # print(scale.shape)

        # if t[0] < 50:
        #     scale = 2
        # else:
        #     scale = 20
        mask = model_kwargs['y']['mask']
        hand_shape = model_kwargs['y']['hand_shape']
        for _ in range(n_guide_steps):
            loss, grad = self.gradients_new(x, hint,hand_shape, mask_hint,mask)
            # print(grad.shape)
            # print(grad[0,:,0,1])
            # print(grad[0,:,0,0])
            grad = model_variance * grad
            # print(grad)
            # print(loss.sum())
            # print(x.shape,grad.shape)
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print(loss.shape)
        log_guide = torch.mean(loss)
        index = t[0].cpu().item()
        self.log_guide_dict[index] = log_guide.cpu().item()
        print(f"{t[0]} step Guide: ",log_guide)
        # print(f"{t[0]} step Guide: ",self.log_guide_dict)
        # with open('/root/code/OmniControl/save/0125_with_global_loss_no_guide/samples_0125_with_global_loss_no_guide_000070000_seed10_predefined/log_guide.pkl','wb')as f:
        #         pickle.dump(self.log_guide_dict,f)
        # # print("Guide: ",loss.shape,grad.shape)
        if index == 0:
            with open('/root/code/OmniControl/save/0125_with_global_loss_no_guide/samples_0125_with_global_loss_no_guide_000070000_seed10_predefined/log_guide_scale2.pkl','wb')as f:
                pickle.dump(self.log_guide_dict,f)
        return x

    def guide_stage2(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        
        # if not train:
        #     scale = self.calc_grad_scale(mask_hint)

        obj_pose = model_kwargs['y']['obj_pose']
        hand_shape = model_kwargs['y']['hand_shape']
        obj_verts = model_kwargs['y']['obj_points']
        for _ in range(n_guide_steps):
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            loss, grad = self.gradients_stage2(x, obj_pose, hint, hand_shape,obj_verts)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        print("Guide: ",torch.mean(loss))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage0(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            # if t[0] < 10:
            #     n_guide_steps = 10
            # else:
            #     n_guide_steps = 2
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach() # hint (bs, nf, 36)
        print(hint)
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0 
        if self.obj_global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.obj_local_mean = self.obj_local_mean.to(hint.device)
            self.obj_local_std = self.obj_local_std.to(hint.device)
            self.obj_global_mean = self.obj_global_mean.to(hint.device)
            self.obj_global_std = self.obj_global_std.to(hint.device)
        hint = hint * self.obj_global_std + self.obj_global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        
        if not train:
            scale = self.calc_grad_scale(mask_hint)

        
        for i in range(n_guide_steps):
            # print(t[0].item(),i)
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            if self.dataset == 'gazehoi_stage0_flag2_lowfps_global' or self.dataset == 'gazehoi_stage0_1obj':
                loss, grad = self.gradients_stage0_global(x,  hint,mask_hint)
            else:
                loss, grad = self.gradients_stage0(x,  hint,mask_hint)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        print("Guide: ",torch.mean(torch.sum(loss,dim=-1)))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage0_1(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 10
            else:
                n_guide_steps = 2
            # if t[0] < 10:
            #     n_guide_steps = 500
            # else:
            #     n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach() # hint (bs, nf, 36)
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0 
        if self.obj_global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.obj_local_mean = self.obj_local_mean.to(hint.device)
            self.obj_local_std = self.obj_local_std.to(hint.device)
            self.obj_global_mean = self.obj_global_mean.to(hint.device)
            self.obj_global_std = self.obj_global_std.to(hint.device)
        hint = hint * self.obj_global_std + self.obj_global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        
        if not train:
            scale = self.calc_grad_scale(mask_hint)

        
        for i in range(n_guide_steps):
            # print(t[0].item(),i)
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            loss, grad = self.gradients_stage0_1(x,  hint,mask_hint)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print("Guide: ",torch.mean(loss))
        # print("Guide: ",loss.shape,grad.shape)
        return x
    
   
    def guide(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # print(t_stopgrad)
        # print(scale)
        # print(min_variance)
        # print(model_kwargs.keys())
        # print(model_kwargs['y'].keys())
        # n_joint = 22 if x.shape[1] == 263 else 21
        n_joint = 17
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        length = model_kwargs['y']['lengths'].clone().detach().long()
        # print(length.shape)
        # print(type(length))
        # print(type(length[0]))
        # length = model_kwargs['y']['length'].clone().detach()
        # print(hint.shape)
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0
        # print(mask_hint.shape)
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        hint = hint * self.global_std + self.global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        # joint id
        # joint_ids = []
        # for m in mask_hint:
        #     joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
        #     joint_ids.append(joint_id)
        
         ##############################################
        if not train:
            scale = self.calc_grad_scale(mask_hint)
        ##################################################
        # if t[0] < 150:
        #     scale = 5
        # else:
        #     scale = 20

        # scale = 1
        # scale = 20
        # if t[0] < 50:
        #     scale = 2
        # else:
        #     scale = 20
        mask = model_kwargs['y']['mask']
        for _ in range(n_guide_steps):
            loss, grad = self.gradients(x, hint, mask_hint,mask,length)
            # print(grad.shape)
            # print(grad[0,:,0,1])
            # print(grad[0,:,0,0])
            grad = model_variance * grad
            # print(grad)
            # print(loss.sum())
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print(loss.shape)
        log_guide = torch.mean(loss)
        index = t[0].cpu().item()
        self.log_guide_dict[index] = log_guide.cpu().item()
        print(f"{t[0]} step Guide: ",log_guide)
        # print(f"{t[0]} step Guide: ",self.log_guide_dict)
        # with open('/root/code/OmniControl/save/0125_with_global_loss_no_guide/samples_0125_with_global_loss_no_guide_000070000_seed10_predefined/log_guide.pkl','wb')as f:
        #         pickle.dump(self.log_guide_dict,f)
        # # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_new(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        n_joints = 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        # bs,nf,_,_ = 
        # print(hint.shape)
        mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(dim=-1, keepdim=True) != 0
        # print(mask_hint.shape)
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        hint = hint.view(hint.shape[0], hint.shape[1],n_joints,3) * mask_hint
        # joint id
        # joint_ids = []
        # for m in mask_hint:
        #     joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
        #     joint_ids.append(joint_id)
        
        ##############################################
        # if not train:
        #     scale = self.calc_grad_scale_kp(mask_hint)
        ##################################################
        if t[0] < 150:
            scale = 5
        else:
            scale = 10

        # print(scale.shape)

        # if t[0] < 50:
        #     scale = 2
        # else:
        #     scale = 20
        mask = model_kwargs['y']['mask']
        hand_shape = model_kwargs['y']['hand_shape']
        for _ in range(n_guide_steps):
            loss, grad = self.gradients_new(x, hint,hand_shape, mask_hint,mask)
            # print(grad.shape)
            # print(grad[0,:,0,1])
            # print(grad[0,:,0,0])
            grad = model_variance * grad
            # print(grad)
            # print(loss.sum())
            # print(x.shape,grad.shape)
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print(loss.shape)
        log_guide = torch.mean(loss)
        index = t[0].cpu().item()
        self.log_guide_dict[index] = log_guide.cpu().item()
        print(f"{t[0]} step Guide: ",log_guide)
        # print(f"{t[0]} step Guide: ",self.log_guide_dict)
        # with open('/root/code/OmniControl/save/0125_with_global_loss_no_guide/samples_0125_with_global_loss_no_guide_000070000_seed10_predefined/log_guide.pkl','wb')as f:
        #         pickle.dump(self.log_guide_dict,f)
        # # print("Guide: ",loss.shape,grad.shape)
        if index == 0:
            with open('/root/code/OmniControl/save/0125_with_global_loss_no_guide/samples_0125_with_global_loss_no_guide_000070000_seed10_predefined/log_guide_scale2.pkl','wb')as f:
                pickle.dump(self.log_guide_dict,f)
        return x

    def guide_stage2(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()
        if self.global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.local_mean = self.local_mean.to(hint.device)
            self.local_std = self.local_std.to(hint.device)
            self.global_mean = self.global_mean.to(hint.device)
            self.global_std = self.global_std.to(hint.device)
        
        # if not train:
        #     scale = self.calc_grad_scale(mask_hint)

        obj_pose = model_kwargs['y']['obj_pose']
        hand_shape = model_kwargs['y']['hand_shape']
        obj_verts = model_kwargs['y']['obj_points']
        for _ in range(n_guide_steps):
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            loss, grad = self.gradients_stage2(x, obj_pose, hint, hand_shape,obj_verts)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        print("Guide: ",torch.mean(loss))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage0(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            # if t[0] < 10:
            #     n_guide_steps = 10
            # else:
            #     n_guide_steps = 2
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach() # hint (bs, nf, 36)
        print(hint)
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0 
        if self.obj_global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.obj_local_mean = self.obj_local_mean.to(hint.device)
            self.obj_local_std = self.obj_local_std.to(hint.device)
            self.obj_global_mean = self.obj_global_mean.to(hint.device)
            self.obj_global_std = self.obj_global_std.to(hint.device)
        hint = hint * self.obj_global_std + self.obj_global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        
        if not train:
            scale = self.calc_grad_scale(mask_hint)

        
        for i in range(n_guide_steps):
            # print(t[0].item(),i)
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            if self.dataset == 'gazehoi_stage0_flag2_lowfps_global' or self.dataset == 'gazehoi_stage0_1obj':
                loss, grad = self.gradients_stage0_global(x,  hint,mask_hint)
            else:
                loss, grad = self.gradients_stage0(x,  hint,mask_hint)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        print("Guide: ",torch.mean(torch.sum(loss,dim=-1)))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def guide_stage0_1(self, x, t, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        # n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 10
            else:
                n_guide_steps = 2
            # if t[0] < 10:
            #     n_guide_steps = 500
            # else:
            #     n_guide_steps = 10

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach() # hint (bs, nf, 36)
        mask_hint = hint.view(hint.shape[0], hint.shape[1],-1).sum(dim=-1, keepdim=True) != 0 
        if self.obj_global_mean.device != hint.device:
            # print(self.global_mean.device, hint.device)
            self.obj_local_mean = self.obj_local_mean.to(hint.device)
            self.obj_local_std = self.obj_local_std.to(hint.device)
            self.obj_global_mean = self.obj_global_mean.to(hint.device)
            self.obj_global_std = self.obj_global_std.to(hint.device)
        hint = hint * self.obj_global_std + self.obj_global_mean # 恢复成未标准化的joint
        hint = hint.view(hint.shape[0], hint.shape[1], -1) * mask_hint
        
        if not train:
            scale = self.calc_grad_scale(mask_hint)

        
        for i in range(n_guide_steps):
            # print(t[0].item(),i)
            # x, obj_pose, hint, hand_shape=None, obj_verts=None
            loss, grad = self.gradients_stage0_1(x,  hint,mask_hint)
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        # print("Guide: ",torch.mean(loss))
        # print("Guide: ",loss.shape,grad.shape)
        return x

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        x = x.requires_grad_()
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        out['mean'] = self.guide_0724_DSG(out, x, t, model_kwargs=model_kwargs, train=True)

        noise = th.randn_like(x)
        # print('const_noise', const_noise)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # print('mean', out["mean"].shape, out["mean"])
        # print('log_variance', out["log_variance"].shape, out["log_variance"])
        # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)


    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
