# --*-- coding:utf-8 --*--
import os, sys, time, random, cv2 
import numpy as np 
import torch 
import os.path as osp
from config import config 
from util import util
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils 
import cv2
from scipy.spatial.distance import cdist
from easydict import EasyDict as edict 
from dataloader import DDDataset, SV_augmentation


class DDDatasetRAM(DDDataset):
    def __init__(self, data_roots=('/media/nlpr/zwzhou/数据资源/MOT16/train',), 
                    subsets = (('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'),),
                    transform=False):
        super(DDDatasetRAM, self).__init__(data_roots, subsets, transform)
        self.K = 8

    def __getitem__(self, index):
        for i, c in enumerate(self.acc_indices):
            if index >= c:
                file_index = index - c
                subset_idx = i   
        pair = self._get_group(self.img_files[subset_idx], self.labels[subset_idx], file_index)
        if self.transform:
            pair = self._transform(pair)
        pair = self._target(pair)
        return pair

    def _get_group(self, img_files, labels, index):
        """ 获取模板和搜索区域对 
        :param img_files:       img 的路径list
        :param labels:          dict, 每帧图像中对应的目标
        :param index:           选择的模板帧索引
        """
        exemplar_img_path = img_files[index]
        exemplar_label_idx = exemplar_img_path.split('/')[-1].split('.')[0]
        while (not (exemplar_label_idx in labels)) or len(labels[exemplar_label_idx]) == 0:
            index = np.random.choice(range(len(img_files)))
            exemplar_img_path = img_files[index]
            exemplar_label_idx = exemplar_img_path.split('/')[-1].split('.')[0]
        
        instance_label_idx = exemplar_label_idx
        instance_img_path = exemplar_img_path
        for _ in range(10):
            tmp_idx = '%06d'%(int(exemplar_label_idx) + np.random.choice(np.arange(-20, 20)))
            if tmp_idx in labels and len(set(labels[tmp_idx][:, 1]) & set(labels[exemplar_label_idx][:, 1])) > 0:
                instance_img_path = instance_img_path.replace(instance_label_idx, tmp_idx)
                instance_label_idx = tmp_idx
                break 
        # 选择tid在两帧中对应的目标box
        track_ids = set(labels[instance_label_idx][:, 1]) & set(labels[exemplar_label_idx][:, 1])
        track_id = random.choice(list(track_ids))
        exemplar_box = labels[exemplar_label_idx][labels[exemplar_label_idx][:, 1] == track_id, 2:][0]  # cxcywh
        instance_box = labels[instance_label_idx][labels[instance_label_idx][:, 1] == track_id, 2:][0] 
        other_boxes = labels[instance_label_idx][labels[instance_label_idx][:, 1] != track_id, 2:]

        img2 = cv2.imread(instance_img_path)
        history_exemplar_img_path, history_exemplar_boxes = [exemplar_img_path], [exemplar_box]
        for i in range(self.K-1):
            tmp_idx = '%06d'%(int(exemplar_label_idx) + np.random.choice(np.arange(-20, 20)))
            if tmp_idx in labels and track_id in labels[tmp_idx][:, 1]:
                history_exemplar_img_path.append(exemplar_img_path.replace(exemplar_label_idx, tmp_idx))
                history_exemplar_boxes.append(labels[tmp_idx][labels[tmp_idx][:, 1]==track_id, 2:][0])
        
        imgs = [cv2.imread(hp) for hp in history_exemplar_img_path]
                
        return self._get_group_samples(imgs, history_exemplar_boxes, img2, instance_box, other_boxes)

    def _get_group_samples(self, imgs, other_boxes, img2, box2, boxes):
        exemplar_imgs = []
        for i in range(self.K):
            j = i%len(imgs)
            img_mean = np.mean(imgs[j], axis=(0,1))
            exemplar_img, _,_,_,_ = self.get_exemplar_image(imgs[j], other_boxes[j],
                            config.exemplar_img_size, config.context, img_mean)
            exemplar_imgs.append(exemplar_img)

        cx, cy, w, h = box2 
        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)

        s_x = s_z / (config.instance_img_size//2)
        img_mean_d = tuple(map(int, img2.mean(axis=(0, 1))))

        a_x_ = np.random.choice(range(-12,12))
        a_x = a_x_ * s_x

        b_y_ = np.random.choice(range(-12,12))
        b_y = b_y_ * s_x  # 让搜索区域的目标发生一定程度的偏移

        instance_img, a_x, b_y, w_x, h_x, _, boxes = self.get_instance_image(img2, box2, boxes.copy(),
                                                                    config.exemplar_img_size, # 127
                                                                    config.instance_img_size,# 255
                                                                    config.context,           # 0.5
                                                                    a_x, b_y,
                                                                    img_mean_d)
        size_x = config.instance_img_size

        x1, y1 = int((size_x + 1) / 2 +a_x- w_x / 2), int((size_x + 1) / 2 + b_y- h_x / 2)
        x2, y2 = int((size_x + 1) / 2 +a_x + w_x / 2), int((size_x + 1) / 2 + b_y + h_x / 2)
        w  = x2 - x1
        h  = y2 - y1
        return {'exemplar_imgs': exemplar_imgs,
                'instance_img': instance_img,
                'relative_cxcywh': [int(a_x), int(b_y), w, h],
                'other_cxcywhs': boxes}

    def _transform(self, pair):
        pair['instance_img'] = SV_augmentation(pair['instance_img'])
        for i in range(self.K):
            pair['exemplar_imgs'][i] = SV_augmentation(pair['exemplar_imgs'][i])
        return pair 

def collate_fn_ram(batch):
    imgs1, imgs2, reg_targets, cls_targets, det_targets, hnw_targets = [], [], [], [], [], []
    for pair in batch:
        imgs1.append(torch.from_numpy(np.stack(pair['exemplar_imgs']).astype(np.float32)).permute(0, 3, 1, 2))
        imgs2.append(torch.from_numpy(pair['instance_img'].astype(np.float32)).permute(2, 0, 1))
        reg_targets.append(torch.from_numpy(pair['reg_target']))
        cls_targets.append(torch.from_numpy(pair['conf_target']).long())
        det_targets.append(torch.from_numpy(pair['score_target']).long())
        hnw_targets.append(torch.from_numpy(pair['hard_neg_weights']))
    return torch.cat(imgs1, dim=0), torch.stack(imgs2, dim=0), torch.stack(reg_targets, dim=0),\
        torch.stack(cls_targets, dim=0), torch.stack(det_targets, dim=0), torch.stack(hnw_targets, dim=0)             