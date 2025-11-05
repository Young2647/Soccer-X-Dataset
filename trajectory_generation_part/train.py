import os, sys

# Fix for cloud training system path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../'))

import torch
import torch.optim as optim
from tqdm import tqdm
from dataloader.dataloader import Soccer_Dataset
from torch.utils.data import DataLoader
from model.network import TrajDiffusion
from config.config import get_args
import wandb

def train_net(args):
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    train_set = Soccer_Dataset(args, model='train')
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=1)
    net = TrajDiffusion(args=args, device=device, style_num=len(train_set.style_set)).to(device)
        
    if args.load_path:
        net.load_state_dict(torch.load(args.load_path))
        print('Model loaded from {}'.format(args.load_path))

    optimizer = optim.AdamW(net.parameters(),lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=args.weight_decay,)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Device: {}
    '''.format(args.epochs, args.batch_size, args.lr, len(train_set), device))

    for epoch in range(args.epochs):
        total_loss = 0
        total_loss_recon = 0
        total_loss_vel = 0

        net.train()

        for data in tqdm(train_dataloader):
            outputs = net.diffusion_train(data)
            
            loss, loss_dict = net.get_loss(data, outputs)
            total_loss += loss
            total_loss_recon += loss_dict['loss_recon']
            total_loss_vel += loss_dict['loss_vel']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wandb_info = {'epoch': epoch, 
                      'loss': total_loss, 
                      'loss_recon': total_loss_recon,
                      'loss_vel': total_loss_vel}
        wandb.log(wandb_info)

        if epoch % args.save_interval == 0:
            os.makedirs(os.path.join(args.save_path, args.name, 'ckpt'), exist_ok=True)
            torch.save(net.state_dict(), os.path.join(args.save_path, args.name, 'ckpt', 'ckpt_' + str(epoch).zfill(6) + '.pth'))
            print('Checkpoint {} saved ! \n'.format(epoch))

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(os.path.join(dir_path, args.save_path, args.name)):
        os.mkdir(os.path.join(dir_path, args.save_path, args.name))
    wandb_mode = 'disabled' if 'debug' in args.name else 'online'
    wandb.init(project='Trajectory Generation Part', name=args.name, config=args, mode=wandb_mode)
    train_net(args)