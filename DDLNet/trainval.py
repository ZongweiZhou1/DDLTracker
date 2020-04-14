# --*-- coding:utf-8 --*--
import os, sys, time, random, logging, argparse
import torch
import numpy as np 
from tqdm import tqdm 
from torch.nn import init 
from torch.utils.data import DataLoader
from config import config 
from DDNet import DDNet
from util import AverageMeter, SavePlot
from dataloader import DDDataset, collate_fn
from dataloader_ram import DDDatasetRAM, collate_fn_ram
torch.manual_seed(1234)

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Net = DDNet(device)
    Net.to(device)

    train_data_roots = ('/media/nlpr/zwzhou/数据资源/MOT16/train', '/media/nlpr/zwzhou/数据资源/MOT15/train')
    train_subsets = (('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-10', 'MOT16-11', 'MOT16-13'),
                    ('ADL-Rundle-6', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
                    'KITTI-17', 'PETS09-S2L1', 'TUD-Stadtmitte', 'Venice-2'))
    mot_dataset = DDDataset(train_data_roots, train_subsets)
    mot_dataloader = DataLoader(mot_dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=True,
                    num_workers=32, drop_last=True, collate_fn=collate_fn)
    
    # resume
    if args.ckp is not None:
        assert os.path.isfile(args.ckp)
        Net.load_state_dict(torch.load(args.ckp, map_location='cpu'))
        torch.cuda.empty_cache()
        print('Train net from: %s'%args.ckp)

    if args.val_data_root is not None:
        val_data_roots = ('/media/nlpr/zwzhou/数据资源/MOT16/train', '/media/nlpr/zwzhou/数据资源/MOT15/train')
        val_subsets = (('MOT16-09', ), ('TUD-Campus', 'ADL-Rundle-8'))
        val_dataset = DDDataset(val_data_roots, val_subsets, transform=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=False, pin_memory=True,
                        num_workers=16, drop_last=False, collate_fn=collate_fn)
        val_avg_loss, val_avg_closs, val_avg_rloss, val_avg_dloss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        train_val_plot = SavePlot('debug', 'train_val_plot')

    avg_loss, avg_closs, avg_rloss, avg_dloss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()   

    print('Training without RAMModule ...')
    for epoch in range(config.epoches):
        Net.train()
        print('Train epoch {}/{}'.format(epoch+1, config.epoches))
        train_loss = []
        with tqdm(total=len(mot_dataloader)) as progbar:
            for _, batch in enumerate(mot_dataloader):
                closs, rloss, dloss, loss = Net.step(batch)
                closs_ = closs.cpu().item()
                if np.isnan(closs_): sys.exit(0)

                avg_closs.update(closs.cpu().item())
                avg_rloss.update(rloss.cpu().item())
                avg_dloss.update(dloss.cpu().item())
                avg_loss.update(loss.cpu().item())
                progbar.set_postfix(closs='{:05.5f}'.format(avg_closs.avg),
                                    rloss='{:05.5f}'.format(avg_rloss.avg),
                                    sloss='{:05.5f}'.format(avg_dloss.avg),
                                    tloss='{:05.3f}'.format(avg_loss.avg))

                progbar.update()
                train_loss.append(avg_loss.avg)
        train_loss = np.mean(train_loss)
        if args.val_data_root is not None:
            # 验证集
            Net.eval()
            val_loss = []
            print('Validation epoch {}/{}'.format(epoch+1, config.epoches))
            with tqdm(total=len(val_dataloader)) as progbar:
                for _, batch in enumerate(val_dataloader):
                    closs, rloss, dloss, loss = Net.step(batch, train=False)
                    closs_ = closs.cpu().item()
                    if np.isnan(closs_): sys.exit(0)

                    val_avg_closs.update(closs.cpu().item())
                    val_avg_rloss.update(rloss.cpu().item())
                    val_avg_dloss.update(dloss.cpu().item())
                    val_avg_loss.update(loss.cpu().item())
                    progbar.set_postfix(closs='{:05.5f}'.format(val_avg_closs.avg),
                                        rloss='{:05.5f}'.format(val_avg_rloss.avg),
                                        sloss='{:05.5f}'.format(val_avg_dloss.avg),
                                        tloss='{:05.3f}'.format(val_avg_loss.avg))

                    progbar.update()
                    val_loss.append(val_avg_loss.avg)
            val_loss = np.mean(val_loss)

            train_val_plot.update(train_loss, val_loss)
            print ('Train loss: {}, val loss: {}'.format(train_loss, val_loss))
    '''save model'''
    Net.save(Net, 'weights', 100)

def main_ram(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Net = DDNet(device)
    Net.to(device)

    train_data_roots = ('/media/nlpr/zwzhou/数据资源/MOT16/train', '/media/nlpr/zwzhou/数据资源/MOT15/train')
    train_subsets = (('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-10', 'MOT16-11', 'MOT16-13'),
                    ('ADL-Rundle-6', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
                    'KITTI-17', 'PETS09-S2L1', 'TUD-Stadtmitte', 'Venice-2'))
    mot_dataset_ram = DDDatasetRAM(train_data_roots, train_subsets)
    mot_dataloader_ram = DataLoader(mot_dataset_ram, batch_size=config.train_batch_size, shuffle=True, pin_memory=True,
                        num_workers=16, drop_last=True, collate_fn=collate_fn_ram)

    # resume
    if args.ckp is not None:
        assert os.path.isfile(args.ckp)
        Net.load_state_dict(torch.load(args.ckp, map_location='cpu'))
        torch.cuda.empty_cache()
        print('Train net from: %s'%args.ckp)

    if args.val_data_root is not None:
        val_data_roots = ('/media/nlpr/zwzhou/数据资源/MOT16/train', '/media/nlpr/zwzhou/数据资源/MOT15/train')
        val_subsets = (('MOT16-09', ), ('TUD-Campus', 'ADL-Rundle-8'))
        val_dataset = DDDatasetRAM(val_data_roots, val_subsets, transform=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config.train_batch_size, shuffle=False, pin_memory=True,
                        num_workers=16, drop_last=False, collate_fn=collate_fn_ram)
        val_avg_loss, val_avg_closs, val_avg_rloss, val_avg_dloss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        train_val_plot = SavePlot('debug', 101)
    avg_loss, avg_closs, avg_rloss, avg_dloss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    print('Training with RAMModule ...')
    
    for epoch in range(config.epoches):
        Net.train()
        train_loss = []
        print('Train epoch {}/{}'.format(epoch+1, config.epoches))
        with tqdm(total=len(mot_dataloader_ram)) as progbar:
            for i, batch in enumerate(mot_dataloader_ram):
                closs, rloss, dloss, loss = Net.step(batch, train_with_ram=True)
                closs_ = closs.cpu().item()
                if np.isnan(closs_): sys.exit(0)

                avg_closs.update(closs.cpu().item())
                avg_rloss.update(rloss.cpu().item())
                avg_dloss.update(dloss.cpu().item())
                avg_loss.update(loss.cpu().item())
                progbar.set_postfix(closs='{:05.3f}'.format(avg_closs.avg),
                                    rloss='{:05.5f}'.format(avg_rloss.avg),
                                    sloss='{:05.5f}'.format(avg_dloss.avg),
                                    tloss='{:05.3f}'.format(avg_loss.avg))

                progbar.update()
                train_loss.append(avg_loss.avg)
    
        train_loss = np.mean(train_loss)
        if args.val_data_root is not None:
            # 验证集
            Net.eval()
            val_loss = []
            print('Validation epoch {}/{}'.format(epoch+1, config.epoches))
            with tqdm(total=len(val_dataloader)) as progbar:
                for _, batch in enumerate(val_dataloader):
                    closs, rloss, dloss, loss = Net.step(batch, train=False, train_with_ram=True)
                    closs_ = closs.cpu().item()
                    if np.isnan(closs_): sys.exit(0)

                    val_avg_closs.update(closs.cpu().item())
                    val_avg_rloss.update(rloss.cpu().item())
                    val_avg_dloss.update(dloss.cpu().item())
                    val_avg_loss.update(loss.cpu().item())
                    progbar.set_postfix(closs='{:05.5f}'.format(val_avg_closs.avg),
                                        rloss='{:05.5f}'.format(val_avg_rloss.avg),
                                        sloss='{:05.5f}'.format(val_avg_dloss.avg),
                                        tloss='{:05.3f}'.format(val_avg_loss.avg))

                    progbar.update()
                    val_loss.append(val_avg_loss.avg)
            val_loss = np.mean(val_loss)

            train_val_plot.update(train_loss, val_loss)
            print ('Train loss: {}, val loss: {}'.format(train_loss, val_loss))
    '''save model'''
    Net.save(Net, 'weights', 'epoch100_ram')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDNet training')
    parser.add_argument('--data_root', default='/media/nlpr/zwzhou/数据资源/MOT16/train', type=str, help='DIR of data')
    parser.add_argument('--val_data_root', default='/media/nlpr/zwzhou/数据资源/MOT16/train', type=str, help='DIR of data')
    parser.add_argument('-ckp', default=None, type=str, help='resume')
    args = parser.parse_args()
    # args.val_data_root = None
    main(args)
    main_ram(args)


