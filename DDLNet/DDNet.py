# --*-- coding:utf-8 --*--
from __future__ import absolute_import, division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from config import config
from losses import rpn_cross_entropy, rpn_smoothL1 
import random
import torch.optim as optim 
from torch.optim.lr_scheduler import ExponentialLR
from collections import namedtuple
import torch.nn.init as init 
import os 


class RAMModule(nn.Module):
    def __init__(self, nh=8, k=4):
        super(RAMModule, self).__init__()
        self.k = k 
        self.nh = nh
        self.std_x, self.std_y = self.k/2, self.k
        self.gaussian_w = self.create_gaussian_w()

        self.relation_layer = nn.Sequential(nn.Linear(self.nh**2, 32),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(32, 32),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(32, self.nh))

    def forward(self, okernels):
        n, c, h, w = okernels.size()
        s_okernels = okernels.view(-1, self.nh, c, h, w)
        update_kernels = []
        for okernels in s_okernels:
            kernels = okernels * self.gaussian_w.to(okernels.device).view(1, 1, self.k, self.k)
            resh_ks = F.normalize(F.max_pool2d(kernels, (self.k, self.k)).squeeze(-1).squeeze(-1), dim=1)
            cor_mat = torch.sum(resh_ks.unsqueeze(0)*resh_ks.unsqueeze(1), dim=-1).view(1, -1)
            resh_kw = F.normalize(self.relation_layer(cor_mat), dim=1).view(self.nh, 1, 1, 1)
            update_kernel = torch.sum(resh_kw*okernels, dim=0, keepdim=True)
            update_kernels.append(update_kernel)
        return torch.stack(update_kernels, dim=0)
        
    def create_gaussian_w(self):
        d = (np.arange(self.k, dtype=np.float32) - self.k/2.0)**2
        x = np.exp(-d/self.std_x)
        y = np.exp(-d/self.std_y)
        w = y[:, np.newaxis] * x[np.newaxis]
        return torch.from_numpy(w)

class DDNet(nn.Module):
    def __init__(self, device=torch.device('cpu'), anchor_num=config.anchor_num):
        super(DDNet, self).__init__()
        self.device = device 
        self.anchor_num = anchor_num
        self.K = 8
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 256, 3, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True))


        # cls branch
        self.cls_conv = nn.Sequential(
             nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2*anchor_num, 1, 1)
        )

        # reg branch
        self.reg_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 1, 1, 0),
            # nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 4*anchor_num, 1, 1)
        )
        
        # det branch
        self.det_conv = nn.Sequential(
                            nn.Conv2d(256, 512, 4, 1),
                            nn.BatchNorm2d(512, eps=1e-6, momentum=0.05),
                            nn.ReLU(inplace=True),
                            # nn.Conv2d(1024, 512, 3, 1, 1),
                            # nn.BatchNorm2d(512, eps=1e-6, momentum=0.05),
                            # nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, 1, 1),
                            nn.BatchNorm2d(512, eps=1e-6, momentum=0.05),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 2*anchor_num, 1, 1, 0)
                            )
        # relationship attention module
        self.ram = RAMModule(self.K)
        self.optimizer = optim.SGD(self.parameters(), lr=config.initial_lr,
                             weight_decay=config.weight_decay, momentum=config.momentum)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=config.lr_decay)
        self.anchors = self.create_anchors().to(device)
        self._initialize_weights()

    def _initialize_weights(self, pretrained_weights=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.0001)
                nn.init.normal_(m.bias.data, std=0.0001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def extract_kernel(self, z):
        return self.feature(z)
    
    def extract_kernel_ram(self, zs):
        kernels = self.extract_kernel(zs)
        return self.ram(kernels)
    
    def inference(self, kernels, x):
        x_patch = self.feature(x)
        N, C, H, W = x_patch.size()
        h, w = kernels.size(-2), kernels.size(-1)
        # conv2d
        z = F.conv2d(x_patch.view(1, N*C, H, W), kernels.view(N*C, 1, h, w), groups=N*C)
        z = z.view(N, C, z.size(-2), z.size(-1))
        # cls
        output_cls = self.cls_conv(z)
        # reg
        output_reg = self.reg_conv(z)
        # det
        output_det = self.det_conv(x_patch)
        return output_reg, output_cls, output_det 

    def forward(self, z, x, train_with_ram=False):
        kernels = self.extract_kernel(z)
        if train_with_ram:
            kernels = self.ram(kernels)
        x_patch = self.feature(x)
        N, C, H, W = x_patch.size()
        h, w = kernels.size(-2), kernels.size(-1)
        # conv2d
        z = F.conv2d(x_patch.view(1, N*C, H, W), kernels.view(N*C, 1, h, w), groups=N*C)
        z = z.view(N, C, z.size(-2), z.size(-1))
        # cls
        output_cls = self.cls_conv(z)
        # reg
        output_reg = self.reg_conv(z)
        # det
        output_det = self.det_conv(x_patch)
        return output_reg, output_cls, output_det 

    def step(self, batch, train=True, train_with_ram=False):
        if train:
            self.train()
        else:
            self.eval()
        z = batch[0].to(self.device)
        x = batch[1].to(self.device)
        reg_target = batch[2].to(self.device)
        cls_target = batch[3].to(self.device)
        det_target = batch[4].to(self.device)
        hnw_weight = batch[5].to(self.device)

        pred_reg, pred_cls, pred_det = self.forward(z, x, train_with_ram)

        pred_cls = pred_cls.reshape(-1, 2, config.total_anchor_num).permute(0, 2, 1)
        pred_reg = pred_reg.reshape(-1, 4, config.total_anchor_num).permute(0, 2, 1)
        pred_det = pred_det.reshape(-1, 2, config.total_anchor_num).permute(0, 2, 1)

        reg_loss = rpn_smoothL1(pred_reg, reg_target, cls_target, self.anchors, config.r_pos, config.beta)
        cls_loss = rpn_cross_entropy(pred_cls, cls_target, config.r_pos, config.r_neg, hnw_weight)
        det_loss = rpn_cross_entropy(pred_det, det_target, config.r_pos, config.r_neg)

        loss = cls_loss + reg_loss + det_loss

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), config.clip)
            self.optimizer.step()
        return cls_loss, reg_loss, det_loss, loss 

    def create_anchors(self):
        '''构建anchors, 
        :return anchors:  response_sz * response * anchor_num x 4, 每个anchor是在原始空间的尺度 
        '''
        response_sz = config.score_size
        anchor_num = config.anchor_num
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)
        size = config.stride**2
        ind = 0
        for ratio in config.anchor_ratios:
            w = int(np.sqrt(size/ratio))
            h = int(w*ratio)
            for scale in config.anchor_scales:
                anchors[ind, 0], anchors[ind, 1] = 0, 0
                anchors[ind, 2], anchors[ind, 3] = w * scale, h*scale 
                ind += 1
        anchors = np.tile(anchors, response_sz**2).reshape((-1, 4))
        begin = -(response_sz // 2) * config.stride
        xs, ys = np.meshgrid(
            begin + config.stride * np.arange(response_sz),
            begin + config.stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return torch.from_numpy(anchors)

    '''save model'''
    def save(self,model, exp_name_dir, epoch):
        model_save_dir_pth = '{}/model'.format(exp_name_dir)
        if not os.path.exists(model_save_dir_pth):
                os.makedirs(model_save_dir_pth)
        net_path = os.path.join(model_save_dir_pth, 'model_e%d.pth' % (epoch + 1))
        torch.save(model.state_dict(), net_path)

if __name__ == '__main__':
    x = torch.rand(1, 3, 271, 271).cuda()
    z = torch.rand(8, 3, 127, 127).cuda()
    net = DDNet().cuda()
    cls, reg, det = net(z, x, True)
    print(cls.size(), reg.size(), det.size())

