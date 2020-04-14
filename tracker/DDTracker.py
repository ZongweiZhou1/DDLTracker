# --*-- coding:utf-8 --*--
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, OrderedDict, deque
from easydict import EasyDict as edict
import os, sys
from DDLNet.config import config
import cv2 

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

    def extract_kernel(self, z):
        return self.feature(z)

    def extract_kernel_ram(self, zs):
        kernels = self.extract_kernel(zs)
        return self.ram(kernels)

    def inference(self, kernels, x):
        x_patch = self.feature(x)
        N, C, H, W = x_patch.size()
        kernels = kernels.expand(N, 1, 1, 1)
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
    

def create_anchors():
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

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
anchors = create_anchors().to(device)
dd_net = DDNet(device).eval()
hann_window = np.outer(np.hanning(config.score_size), np.hanning(config.score_size))
hann_window = np.tile(hann_window.flatten(), config.anchor_num)


class DDTracker(object):
    def __init__(self):
        super(DDTracker, self).__init__()
        self.kernel = None
        self.exemplar_patches = []  # 用于更新kernel

    def init(self, image, tlwh):
        image = np.asarray(image)
        avg_color = np.asarray(image)
        center, target_sz = tlwh[:2][::-1] + tlwh[2:][::-1]/2.0, tlwh[2:][::-1]
        # exemplar and search sizes
        context = config.context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * config.instance_size / config.exemplar_size
        # exemplar image
        
        exemplar_image = self._crop_and_resize(image, center, z_sz, config.exemplar_size, avg_color)
        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.no_grad():
            self.kernel = dd_net.extract_kernel(exemplar_image)
        self.center = center
        self.x_sz = x_sz 
        self.z_sz = z_sz
        self.target_sz = target_sz
        self.avg_color = avg_color

    def predict(self, image):
        image = np.asarray(image)
        instance_image = self._crop_and_resize(image, self.center, self.x_sz, 
                                config.instance_img_size, self.avg_color)
        instance_image = torch.from_numpy(instance_image).to(device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.no_grad():
            pred_reg, pred_cls, pred_det = dd_net.inference(self.kernel, instance_image)
        
        # offsets 解码中心偏置量
        offsets = pred_reg.permute(1,2,3,0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * anchors[:, 2] + anchors[:, 0]
        offsets[1] = offsets[1] * anchors[:, 3] + anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * anchors[:, 3]
        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets, self.z_sz)
        cls_response = F.softmax(pred_cls.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = cls_response * penalty
        det_response = F.softmax(pred_det.permute(1,2,3,0).contiguous().view(2,-1), dim=0).data[1].cpu().numpy()
        response = response*det_response
        response = (1 - config.window_influence) * response + \
                    config.window_influence * hann_window
        # 找到响应最大的anchor, 这里的响应满足三点: 离中心的距离, 月kernel的匹配度,st包含行人的概率
        best_id = np.argmax(response)
        offset = offsets[:, best_id] * self.z_sz/config.exemplar_size
        
        # 当前轨迹预测的中心点
        center = np.clip(self.center + offset[:2][::-1], 0, image.shape[:2])
        # 当前轨迹预测的目标尺寸
        lr = response[best_id] * config.scale_lr
        target_sz = np.clip((1-lr) * self.target_sz + lr *(offset[2:][::-1]), 10, image.shape[:2])

        tlwh = np.array([center[1] - (target_sz[1]-1)/2.0, center[0] - (target_sz[0]-1)/2.0,
                        target_sz[1], target_sz[0]])
        return tlwh

    def update(self, tlwh, image=None):
        image = np.asarray(image)
        avg_color = np.asarray(image)
        center, target_sz = tlwh[:2][::-1] + tlwh[2:][::-1]/2.0, tlwh[2:][::-1]
        # exemplar and search sizes
        context = config.context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * config.instance_size / config.exemplar_size
        self.center = center
        self.x_sz = x_sz 
        self.z_sz = z_sz 
        self.target_sz = target_sz 
        if image is not None:
            exemplar_image = self._crop_and_resize(image, center, z_sz, config.exemplar_size, avg_color)
            self.exemplar_patches.append(exemplar_image)
            if len(self.exemplar_patches) > 16:
                example_images = torch.from_numpy(np.stack(self.exemplar_patches[-15::2], axis=0)).permute(0, 2,3,1).float()
                self.kernel = dd_net.extract_kernel_ram(example_images)
                self.exemplar_patches = self.exemplar_patches[-8:]


    def matching(self, image, tlwhs):
        image = np.asarray(image)
        avg_color = np.mean(image, axis=(0,1))
        instance_images, target_szs, z_szs = [], [], []
        for tlwh in tlwhs:
            center, target = tlwh[:2][::-1] + tlwh[2:][::-1]/2.0, tlwh[2:][::-1]
            # exemplar and search sizes
            context = config.context * np.sum(target_sz)
            z_sz = np.sqrt(np.prod(target_sz + context))
            x_sz = z_sz * config.instance_size / config.exemplar_size
            instance_images.append(self._crop_and_resize(image, center, x_sz, 
                                config.instance_img_size, avg_color))
            target_szs.append(target_sz)
            z_szs.append(z_sz)

        instance_images = np.stack(instance_images, axis=0)
        instance_image = torch.from_numpy(instance_image).to(device).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            pred_regs, pred_clss, pred_dets = dd_net.inference(self.kernel, instance_image)
        scores = []
        for i in range(len(tlwhs)):
            # offsets 解码中心偏置量
            pred_reg, pred_cls, pred_det = pred_regs[i], pred_clss[i], pred_dets[i]
            offsets = pred_reg.view(4, -1).cpu().numpy()
            offsets[0] = offsets[0] * anchors[:, 2] + anchors[:, 0]
            offsets[1] = offsets[1] * anchors[:, 3] + anchors[:, 1]
            offsets[2] = np.exp(offsets[2]) * anchors[:, 2]
            offsets[3] = np.exp(offsets[3]) * anchors[:, 3]
            # scale and ratio penalty
            penalty = self._create_penalty(target_szs[i], offsets, z_szs[i])
            cls_response = F.softmax(pred_cls.view(2, -1), dim=0).data[1].cpu().numpy()
            response = cls_response * penalty
            det_response = F.softmax(pred_det.view(2,-1), dim=0).data[1].cpu().numpy()
            response = response*det_response
            response = (1 - config.window_influence) * response + \
                        config.window_influence * hann_window
            # 找到响应最大的anchor, 这里的响应满足三点: 离中心的距离, 月kernel的匹配度,st包含行人的概率
            scores.append(np.max(response))
        return np.array(scores)

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))
        return patch

    def _create_penalty(self, target_sz, offsets, z_sz):
        '''
        :param target_sz:   目标的hw
        :param offsets:     4 x H x W x anchor_num, 每一个anchor的偏置量
        :param z_sz:        目标截取的尺寸
        :return penalty:    每一个anchor回归框的因尺寸变化导致的惩罚项
        '''
        def padded_size(w,h):
            context = config.context*(w+h)
            return np.sqrt((w + context) * (h + context))
        
        def larger_ratio(r):
            return np.maximum(r, 1/r)

        src_sz = padded_size(*(target_sz * config.exemplar_size / z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz/src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
            config.penalty_k)

        return penalty




