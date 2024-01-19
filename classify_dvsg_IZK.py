# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:54:32 2023

@author: yl5922
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 22:32:54 2023

@author: yl5922
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse

import numpy as np
from IZK_neuron import IZK_neuron

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)
    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class PythonNet(nn.Module):
    def __init__(self, channels: int, hidden_num = 800):
        super().__init__()
        conv = []
        conv.extend(PythonNet.conv3x3(2, channels))
        conv.append(nn.MaxPool2d(2, 2))
        for i in range(4):
            conv.extend(PythonNet.conv3x3(channels, channels))
            conv.append(nn.MaxPool2d(2, 2))
        conv.append(nn.Flatten())
        self.conv = nn.Sequential(*conv)
        
        self.fc1 = nn.Sequential(
                layer.Dropout(0.5),
                nn.Linear(channels * 4 * 4, hidden_num, bias=False),
                neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
                nn.Flatten()
            )
        
        self.fc2 = nn.Sequential(
                layer.Dropout(0.5),
                nn.Linear(hidden_num, 110, bias=False),
                neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
                nn.Flatten()
            )

        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        input_spk_rec = []
        hidden_spk_rec = []        
        
        out_spikes = 0
        for t in range(x.shape[0]):
            input_spk = self.conv(x[t])
            hidden_spk = self.fc1(input_spk)
            out_spk = self.fc2(hidden_spk)
            
            out_spikes += self.vote(out_spk)
            input_spk_rec.append(input_spk)
            hidden_spk_rec.append(hidden_spk)
            
        input_spk_rec = torch.stack(input_spk_rec,dim=1)
        hidden_spk_rec = torch.stack(hidden_spk_rec,dim=1)
        spk_rec = [input_spk_rec, hidden_spk_rec]
        return out_spikes / x.shape[0], spk_rec


    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        ]


class IZKNet(nn.Module):
    def __init__(self, channels: int, hidden_num=800, simplified = False):
        super().__init__()
        conv = []
        conv.extend(IZKNet.conv3x3(2, channels, simplified))
        conv.append(nn.MaxPool2d(2, 2))
        for i in range(4):
            conv.extend(IZKNet.conv3x3(channels, channels, simplified))
            conv.append(nn.MaxPool2d(2, 2))
        conv.append(nn.Flatten())
        self.conv = nn.Sequential(*conv)

        self.fc1 = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(channels * 4 * 4, hidden_num, bias=False),
            IZK_neuron(simplified= simplified, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten()
        )

        self.fc2 = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(hidden_num, 110, bias=False),
            IZK_neuron(simplified= simplified, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten()
        )

        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        input_spk_rec = []
        hidden_spk_rec = []

        out_spikes = 0
        for t in range(x.shape[0]):
            input_spk = self.conv(x[t])
            hidden_spk = self.fc1(input_spk)
            out_spk = self.fc2(hidden_spk)

            out_spikes += self.vote(out_spk)
            input_spk_rec.append(input_spk)
            hidden_spk_rec.append(hidden_spk)

        input_spk_rec = torch.stack(input_spk_rec, dim=1)
        hidden_spk_rec = torch.stack(hidden_spk_rec, dim=1)
        spk_rec = [input_spk_rec, hidden_spk_rec]
        return out_spikes / x.shape[0], spk_rec

    @staticmethod
    def conv3x3(in_channels: int, out_channels, simplified):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            IZK_neuron(simplified= simplified, surrogate_function=surrogate.ATan(), detach_reset=True),
        ]

def main():
    # python classify_dvsg.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024 -IZK
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-channels', default=64, type=int, help='channels of Conv2d in SNN')
    parser.add_argument('-data_dir', default= './DVS128Gesture', type=str, help='root dir of DVS128 Gesture dataset')
    parser.add_argument('-out_dir', default= './dvsglog', type=str, help='root dir for saving logs and checkpoint')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', default =True, action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use CUDA neuron and multi-step forward mode')


    parser.add_argument('-opt', default = 'Adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-reg_weight_input', default = '0', type=float, help='The regular loss on the sparsity of input spikes')
    parser.add_argument('-reg_weight_hidden', default = '0', type=float, help='The regular loss on the sparsity of hidden spikes')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-hidden_num', default=800, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-IZK', action='store_true', default=True, help='Use IZK neurons, if false, use the LIF model')
    parser.add_argument('-simplified', action='store_true', default=True, help='Use simplified IZK neurons, which will significantly enhance the performance')

    args = parser.parse_args()
    print(args)
    
    for outer_num in range(1):
        args.hidden_num = (outer_num+1)*800
        
        
        if args.cupy:
            pass
        else:
            if args.IZK:
                net = IZKNet(channels=args.channels, hidden_num=args.hidden_num, simplified=args.simplified)
            else:
                net = PythonNet(channels=args.channels, hidden_num=args.hidden_num)
        print(net)
        net.to(args.device)

        optimizer = None
        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        else:
            raise NotImplementedError(args.opt)
    
        lr_scheduler = None
        if args.lr_scheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'CosALR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
        else:
            raise NotImplementedError(args.lr_scheduler)
    
        train_set = DVS128Gesture(args.data_dir, train=True, data_type='frame', split_by='number', frames_number=args.T)
        test_set = DVS128Gesture(args.data_dir, train=False, data_type='frame', split_by='number', frames_number=args.T)
    
        train_data_loader = DataLoader(
            dataset=train_set,
            batch_size=args.b,
            shuffle=True,
            num_workers=args.j,
            drop_last=True,
            pin_memory=True)
    
        test_data_loader = DataLoader(
            dataset=test_set,
            batch_size=args.b,
            shuffle=False,
            num_workers=args.j,
            drop_last=False,
            pin_memory=True)
    
        scaler = None
        if args.amp:
            scaler = amp.GradScaler()
    
        start_epoch = 0
        max_test_acc = 0
    
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            max_test_acc = checkpoint['max_test_acc']
    
        out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_c_{args.channels}_{args.opt}_lr_{args.lr}_hidden_{args.hidden_num}_hreg_{args.reg_weight_hidden}_ireg_{args.reg_weight_input}_IZK_{args.IZK}')
        if args.lr_scheduler == 'CosALR':
            out_dir += f'CosALR_{args.T_max}'
        elif args.lr_scheduler == 'StepLR':
            out_dir += f'StepLR_{args.step_size}_{args.gamma}'
        else:
            raise NotImplementedError(args.lr_scheduler)
    
        if args.amp:
            out_dir += '_amp'
        if args.cupy:
            out_dir += '_cupy'
    
    
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print(f'Mkdir {out_dir}.')
    
        with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
    
        writer = SummaryWriter(os.path.join(out_dir, 'dvsg_logs'), purge_step=start_epoch)
    
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            net.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0
            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 11).float()
                if args.amp:
                    with amp.autocast():
                        out_fr, spk_rec = net(frame)
                        input_spk = spk_rec[0]
                        hidden_spk = spk_rec[1]
                        reg_loss = args.reg_weight_input * torch.sum(input_spk) + args.reg_weight_hidden * torch.sum(
                            hidden_spk)  # L1 loss on total number of spikes
                        reg_loss += args.reg_weight_input * torch.mean(
                            torch.sum(torch.sum(input_spk, dim=0), dim=0) ** 2) + args.reg_weight_hidden * torch.mean(
                            torch.sum(torch.sum(hidden_spk, dim=0), dim=0) ** 2)  # L2 loss on spikes per neuron

                        loss = F.mse_loss(out_fr, label_onehot)  + reg_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out_fr, spk_rec = net(frame)
                    input_spk = spk_rec[0]
                    hidden_spk = spk_rec[1]
                    
                    reg_loss = args.reg_weight_input*torch.sum(input_spk) + args.reg_weight_hidden*torch.sum(hidden_spk)# L1 loss on total number of spikes
                    reg_loss += args.reg_weight_input*torch.mean(torch.sum(torch.sum(input_spk,dim=0),dim=0)**2) +  args.reg_weight_hidden*torch.mean(torch.sum(torch.sum(hidden_spk,dim=0),dim=0)**2)# L2 loss on spikes per neuron
                             
                    
                    loss = F.mse_loss(out_fr, label_onehot) + reg_loss
                    loss.backward()
                    optimizer.step()
    
                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()
    
                functional.reset_net(net)
            train_loss /= train_samples
            train_acc /= train_samples
    
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()
    
            net.eval()
            test_loss = 0
            test_acc = 0
            test_samples = 0
            with torch.no_grad():
                input_spc_rec = []
                hidden_spc_rec = []
                for frame, label in test_data_loader:
                    frame = frame.float().to(args.device)
                    label = label.to(args.device)
                    label_onehot = F.one_hot(label, 11).float()
                    out_fr, spk_rec = net(frame)
                    input_spk = spk_rec[0]
                    hidden_spk = spk_rec[1]
                    
                    reg_loss = args.reg_weight_input*torch.sum(input_spk) + args.reg_weight_hidden*torch.sum(hidden_spk)# L1 loss on total number of spikes
                    reg_loss += args.reg_weight_input*torch.mean(torch.sum(torch.sum(input_spk,dim=0),dim=0)**2) +  args.reg_weight_hidden*torch.mean(torch.sum(torch.sum(hidden_spk,dim=0),dim=0)**2)# L2 loss on spikes per neuron
                    
                    loss = F.mse_loss(out_fr, label_onehot) + reg_loss
    
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)
    
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
    
            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                save_max = True
    
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
    
            if save_max:
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
    
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
    
            print(args)
            print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')


if __name__ == '__main__':
        main()
