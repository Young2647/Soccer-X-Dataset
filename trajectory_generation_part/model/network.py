import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class TrajProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.Embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.Embedding(x)  
        return x
    

class DestinationProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.Embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats = x.shape
        x = x.unsqueeze(0)
        x = self.Embedding(x)  
        return x
        
class DirectionProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.Embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats = x.shape
        x = x.unsqueeze(0)
        x = self.Embedding(x)  
        return x


class OpponentProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.Embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats = x.shape
        x = x.unsqueeze(0)
        x = self.Embedding(x)  
        return x    
    
    
class OutputProcess(nn.Module):
    def __init__(self, latent_dim, nfeats):
        super().__init__()
        self.latent_dim = latent_dim
        self.nfeats = nfeats
        self.Embedding = nn.Linear(self.latent_dim, self.nfeats)

    def forward(self, output):
        nframes, bs, latent_dim = output.shape
        output = self.Embedding(output)  
        output = output.reshape(nframes, bs, self.nfeats)
        output = output.permute(1,2,0)
        return output    
    

class EmbedStyle(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input.to(torch.long) 
        output = self.action_embedding[idx]
        return output
    

class TrajDiffusion(nn.Module):
    def __init__(self, args, style_num, nfeats=8, past_frame=10, future_frame=45, device='cpu',
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", cond_mask_prob=0):
        super().__init__()
        
        self.args = args

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.past_frame = past_frame
        self.future_frame = future_frame
        self.device = device
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.future_traj_process = TrajProcess(8, self.latent_dim)
        self.past_traj_process = TrajProcess(8, self.latent_dim)
        self.destination_process = DestinationProcess(8, self.latent_dim)
        self.embed_style = EmbedStyle(style_num, self.latent_dim)


        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers)
        self.output_process = OutputProcess(self.latent_dim, self.nfeats)
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x, past_traj, destination, style_idx):
        bs, nfeats, nframes  = x.shape
        
        future_traj_emb = self.future_traj_process(x)
        past_traj_emb = self.past_traj_process(past_traj)
        destination_emb = self.destination_process(destination)
        style_emb = self.embed_style(style_idx).unsqueeze(0)
        
        xseq = torch.cat((style_emb, 
                          past_traj_emb, destination_emb,
                          future_traj_emb), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:] 
        output = self.output_process(output)  
        return output
        
    def diffuse(self, x_0, cond):
        x_t = torch.randn_like(x_0).permute(0, 2, 1)
        return self.forward(x_t, cond['past_traj'], cond['destination'], cond['style_idx']).permute(0,2,1)

    def diffusion_train(self, datas):
        data = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['data'].items()}
        cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['cond'].items()}
        cond['past_traj'] = torch.cat((cond['traj_trans_past'], cond['traj_poses_past']), dim=-1).permute(0, 2, 1)
        cond['destination'] = torch.cat((cond['dest_trans'], cond['dest_poses']), dim=-1)
        x_0 = torch.cat((data['traj_trans_future'], data['traj_poses_future']), dim=-1)
        return self.diffuse(x_0, cond)
    
    def diffusion_test(self, datas):
        data = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['data'].items()}
        cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['cond'].items()}
        cond['past_traj'] = torch.cat((cond['traj_trans_past'], cond['traj_poses_past']), dim=-1).permute(0, 2, 1)
        cond['destination'] = torch.cat((cond['dest_trans'], cond['dest_poses']), dim=-1)
        x_0 = torch.cat((data['traj_trans_future'], data['traj_poses_future']), dim=-1)
        x_0 = torch.randn_like(x_0)

        with torch.no_grad():
            t = torch.full((x_0.shape[0],), torch.tensor(1.), dtype=torch.long).to(self.device)
            x_0 = self.diffuse(x_0, cond)
        output = x_0
        return output

    def get_loss(self, datas, x_pred):
        data = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['data'].items()}

        x_gt = torch.cat((data['traj_trans_future'], data['traj_poses_future']), dim=-1)
        loss_recon = self.loss_fn(x_gt, x_pred)
        
        vel_pred = x_pred[:, 1:] - x_pred[:, :-1]
        vel_gt = x_gt[:, 1:] - x_gt[:, :-1]
        loss_vel = self.loss_fn(vel_gt, vel_pred)

        loss = loss_recon + loss_vel
        loss_dict = {'loss': loss,
                     'loss_recon': loss_recon,
                     'loss_vel': loss_vel}
        return loss, loss_dict