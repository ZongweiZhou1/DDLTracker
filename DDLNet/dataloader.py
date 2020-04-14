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


DEBUG = False

def SV_augmentation(img):
    fraction = 0.50
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = img_hsv[:, :, 1].astype(np.float32), img_hsv[:, :, 2].astype(np.float32)
    a = (random.random() * 2 - 1) * fraction + 1
    S = S * a
    if a > 1: np.clip(S, a_min=0, a_max=255, out=S)
    a = (random.random() * 2 - 1) * fraction + 1
    V = V * a
    if a > 1: np.clip(V, a_min=0, a_max=255, out=V)
    img_hsv[:, :, 1], img_hsv[:, :, 2] = S.astype(np.uint8), V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img

class DDDataset(Dataset):
    def __init__(self, data_roots=('/media/nlpr/zwzhou/数据资源/MOT16/train',), 
                    subsets = (('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'),),
                    transform=True):
        self.visible_th = 0.4
        self.img_files, self.labels = [], []
        for i, dr in enumerate(data_roots):
            subset = subsets[i]
            for ss in subset:
                self.img_files.append([osp.join(dr, ss, 'img1', k) 
                                        for k in os.listdir(osp.join(dr, ss, 'img1'))])
                self.labels.append(self.read_labels(osp.join(dr, ss, 'gt/gt.txt')))
        self.index_split = [len(t) for t in self.img_files]
        self.acc_indices = [sum(self.index_split[:i]) for i in range(len(self.index_split))]
        self.transform=transform
        self.anchors = util.generate_anchors(config.stride, config.stride,
                                    config.anchor_scales,
                                    config.anchor_ratios, config.score_size)

    def __len__(self):
        return self.acc_indices[-1]+self.index_split[-1]

    def read_labels(self, label_path):
        annos = np.loadtxt(label_path, delimiter=',')
        label_dict = edict()
        for frame_idx in set(annos[:, 0]):
            tmp = annos[annos[:, 0] == frame_idx]
            if tmp.shape[1] == 9:
                tmp = tmp[np.logical_or(tmp[:, -2]==1, tmp[:, -2]==7)]
                tmp = tmp[tmp[:, -1] > self.visible_th]
            elif tmp.shape[1] == 10:
                tmp = tmp[tmp[:, 6] == 1]
            tmp[:, 2:4] += tmp[:, 4:6]/2  # cxcywh 
            label_dict['%06d'%int(frame_idx)] = tmp[:, :6]  # frameid, tid, cxcywh
        return label_dict

    def __getitem__(self, index):
        for i, c in enumerate(self.acc_indices):
            if index >= c:
                file_index = index - c
                subset_idx = i   
        pair = self.get_pairs(self.img_files[subset_idx], self.labels[subset_idx], file_index)
        if self.transform:
            pair = self._transform(pair)
        pair = self._target(pair)
        return pair

    def get_pairs(self, img_files, labels, index):
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
        for i in range(10):
            tmp_idx = '%06d'%(int(exemplar_label_idx) + np.random.choice(np.arange(-20, 20)))
            if tmp_idx in labels and len(set(labels[tmp_idx][:, 1]) & set(labels[exemplar_label_idx][:, 1])) > 0:
                instance_img_path = instance_img_path.replace(instance_label_idx, tmp_idx)
                instance_label_idx = tmp_idx
                break 
        # 选择tid在两帧中对应的目标box
        track_ids = set(labels[instance_label_idx][:, 1]) & set(labels[exemplar_label_idx][:, 1])
        track_id = random.choice(list(track_ids))
        exemplar_box = labels[exemplar_label_idx][labels[exemplar_label_idx][:, 1] == track_id, 2:]  # cxcywh
        instance_box = labels[instance_label_idx][labels[instance_label_idx][:, 1] == track_id, 2:] 
        other_boxes = labels[instance_label_idx][labels[instance_label_idx][:, 1] != track_id, 2:]

        img1 = cv2.imread(exemplar_img_path)
        img2 = cv2.imread(instance_img_path)

        if DEBUG:
            #验证是否对应
            box1, box2 = exemplar_box[0], instance_box[0]
            cv2.imwrite('debug/img_1.jpg', img1[int(box1[1]-box1[3]/2):int(box1[1]+box1[3]/2),
                                            int(box1[0]-box1[2]/2):int(box1[0]+box1[2]/2)])
            cv2.imwrite('debug/img_2.jpg', img2[int(box2[1]-box2[3]/2):int(box2[1]+box2[3]/2),
                                            int(box2[0]-box2[2]/2):int(box2[0]+box2[2]/2)])  
        
        return self._get_pair_samples(img1, exemplar_box[0], img2, instance_box[0], other_boxes)

    def _get_pair_samples(self, img1, box1, img2, box2, boxes):
        """ 从图像中抽取一对样本对
        :param box1, box2: 都是未归一化的cx,cy,w,h
        :param boxes:  来源于img2中的box,有可能出现在box2截取的图中,用于distractor aware的训练
        """
        img1_mean = np.mean(img1, axis=(0,1))
        exemplar_img, scale_z, s_z, w_x, h_x = self.get_exemplar_image(img1, box1,
                                config.exemplar_img_size, config.context, img1_mean)
        size_x = config.exemplar_img_size
        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)
        x2, y2 = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)  # 表示resize之后,目标的位置, 左上角坐标原点
        if DEBUG:
            # 验证crop出的图像块是否对应
            e_img = exemplar_img.copy()
            cv2.rectangle(e_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
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
        if DEBUG:
            # 验证crop出的图像块是否对应
            img = instance_img.copy()
            cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 255), 2)
            for b in boxes:
                x1, y1 = int((size_x + 1) / 2 +b[0]- b[2] / 2), int((size_x + 1) / 2 + b[1]- b[3] / 2)
                x2, y2 = int((size_x + 1) / 2 +b[0] + b[2] / 2), int((size_x + 1) / 2 + b[1] + b[3] / 2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 2)
            # half = int((size_x+1)/2)
            # for anchor in self.anchors[::361]:
            #     wh, hh = int(anchor[2]//2), int(anchor[3]//2)
            #     cv2.rectangle(instance_img, (half-wh, half-hh),(half+wh, half+hh), (0, 255, 0), 2)
            # for anchor in self.anchors[1::361]:
            #     wh, hh = int(anchor[2]//2), int(anchor[3]//2)
            #     ax, by = 8, 8
            #     cv2.rectangle(instance_img, (half-wh+ax, half-hh +by),(half+wh+ax, half+hh+by), (0, 255, 255), 2)

            cv2.imwrite('debug/exemplar_img.jpg', exemplar_img)
            cv2.imwrite('debug/instance_img.jpg', img)

        return {'exemplar_img': exemplar_img,
                'instance_img': instance_img,
                'relative_cxcywh': [int(a_x), int(b_y), w, h],
                'other_cxcywhs': boxes}

    def get_exemplar_image(self, img, bbox, size_z, context_amount, img_mean=None):
        """ crop具有context信息的exemplar图像, size_z目标大小"""
        cx, cy, w, h = bbox

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)  # 加上上下文信息之后的方块
        scale_z = size_z / s_z  #crop的方块需要缩放的比例

        exemplar_img, scale_x = self.crop_and_pad_old(img, cx, cy, size_z, s_z, img_mean)  # scale_x 是最终取整操作之后达到目标尺寸需要缩放比例

        w_x = w * scale_x
        h_x = h * scale_x

        return exemplar_img, scale_z, s_z, w_x, h_x  # scale_z表示缩放比例, s_z表示原始图像块截取的边长, wx, hx分别表示在缩放后的图像中目标的尺寸

    def get_instance_image(self, img, bbox, boxes, size_z, size_x, context_amount, a_x, b_y, img_mean=None):
        """ crop具有context信息的instance图像, size_z, size_x分别表示模板和搜索区域大小, a_x, b_y表示目标在待截取图像中心的相对坐标 
        最终截取的图像时 cx-a_x, cy-b_y为中心的区域
        boxes: np.array, N x 4, (cx,cy,w,h), 默认值 np.empty((0,4))
        """
        cx, cy, w, h = bbox  # float type

        #cx, cy = cx - a_x , cy - b_y
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z) # the width of the crop box

        s_x = s_z * size_x / size_z  # 要截取的区域的大小
        instance_img, gt_w, gt_h, scale_x, scale_h, scale_w = self.crop_and_pad(img, cx, cy, w, h, a_x, b_y,  size_x, s_x, img_mean)
        w_x = gt_w #* scale_x #w * scale_x
        h_x = gt_h #* scale_x #h * scale_x
        if len(boxes) > 0:
            boxes[:, 0] = (boxes[:,0]-(cx-a_x)) * scale_w * scale_x
            boxes[:, 1] = (boxes[:,1]-(cy-b_y)) * scale_h * scale_x
            boxes[:, 2] *= (scale_w *scale_x)
            boxes[:, 3] *= (scale_h *scale_x) # 转换到resize之后的相对位置和尺寸
            # 转换成tlbr 进行筛选
            boxes[:,:2] -= boxes[:, 2:]/2.0
            boxes[:,2:] += boxes[:, :2]
            half_size = config.instance_img_size/2
            mask = np.logical_and(np.logical_and(boxes[:,0]<half_size, boxes[:, 2]>-half_size), 
                        np.logical_and(boxes[:,1]<half_size, boxes[:, 3]>-half_size))
            # 转换成cxcywh
            boxes[:, 2:] -= boxes[:, :2]
            boxes[:, :2] += boxes[:, 2:]/2.0
            boxes = boxes[mask]

        a_x, b_y = a_x*scale_w*scale_x, b_y*scale_h*scale_x  # resize之后目标相对中心点的位置
        #cv2.imwrite('1.jpg', frame)
        return instance_img, a_x, b_y, w_x, h_x, scale_x, boxes

    def crop_and_pad(self, img, cx, cy, gt_w, gt_h, a_x, b_y, model_sz, original_sz, img_mean=None):
        """ cx, cy, gt_w, gt_h 目标框的box,
        a_x, b_y: 中心点的偏移
        model_sz:  目标的尺寸
        original_sz: 原始图像中包含context的crop尺寸
        """

        #random = np.random.uniform(-0.15, 0.15)
        scale_h = 1.0 + np.random.uniform(-0.15, 0.15)  # 一定尺度的缩放
        scale_w = 1.0 + np.random.uniform(-0.15, 0.15)

        im_h, im_w, _ = img.shape

        xmin = (cx-a_x) - ((original_sz - 1) / 2)* scale_w
        xmax = (cx-a_x) + ((original_sz - 1) / 2)* scale_w

        ymin = (cy-b_y) - ((original_sz - 1) / 2)* scale_h
        ymax = (cy-b_y) + ((original_sz - 1) / 2)* scale_h

        #print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)

        left   = int(self.round_up(max(0., -xmin)))
        top    = int(self.round_up(max(0., -ymin)))
        right  = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))

        r, c, k = img.shape
        if any([top, bottom, left, right]):
            # te_im_ = np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)  # 0 is better than 1 initialization
            te_im = np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)  # 0 is better than 1 initialization

            #cv2.imwrite('te_im1.jpg', te_im)
            te_im[:, :, :] = img_mean
            #cv2.imwrite('te_im2_1.jpg', te_im)
            te_im[top:top + r, left:left + c, :] = img
            #cv2.imwrite('te_im2.jpg', te_im)

            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean

            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

            #cv2.imwrite('te_im3.jpg',   im_patch_original)

        else:
            im_patch_original = img[int(ymin):int((ymax) + 1), int(xmin):int((xmax) + 1), :]

            #cv2.imwrite('te_im4.jpg', im_patch_original)

        if not np.array_equal(model_sz, original_sz):

            h, w, _ = im_patch_original.shape


            if h < w:
                scale_h_ = 1
                scale_w_ = h/w
                scale = config.instance_img_size/h
            elif h > w:
                scale_h_ = w/h
                scale_w_ = 1
                scale = config.instance_img_size/w
            elif h == w:
                scale_h_ = 1
                scale_w_ = 1
                scale = config.instance_img_size/w

            gt_w = gt_w * scale_w_
            gt_h = gt_h * scale_h_

            gt_w = gt_w * scale
            gt_h = gt_h * scale

            #im_patch = cv2.resize(im_patch_original_, (shape))  # zzp: use cv to get a better speed
            #cv2.imwrite('te_im8.jpg', im_patch)

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
            #cv2.imwrite('te_im9.jpg', im_patch)


        else:
            im_patch = im_patch_original
        #scale = model_sz / im_patch_original.shape[0]
        return im_patch, gt_w, gt_h, scale, scale_h_, scale_w_  # gt_w, gt_h resize之后图像中目标的大小, scale_h_, scale_w_h, 两个方向上的缩放值

    def crop_and_pad_old(self, img, cx, cy, model_sz, original_sz, img_mean=None):
        im_h, im_w, _ = img.shape

        xmin = cx - (original_sz - 1) / 2
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2
        ymax = ymin + original_sz - 1

        left = int(self.round_up(max(0., -xmin)))
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))
        r, c, k = img.shape
        if any([top, bottom, left, right]):
            te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
            te_im[top:top + r, left:left + c, :] = img
            if top:
                te_im[0:top, left:left + c, :] = img_mean
            if bottom:
                te_im[r + top:, left:left + c, :] = img_mean
            if left:
                te_im[:, 0:left, :] = img_mean
            if right:
                te_im[:, c + left:, :] = img_mean
            im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        else:
            im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        if not np.array_equal(model_sz, original_sz):

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original
        scale = model_sz / im_patch_original.shape[0]
        return im_patch, scale

    def round_up(self, value):
        return round(value + 1e-6 + 1000) - 1000 

    def _target(self, pair):
        reg_target, conf_target, score_target, hard_neg_weights = self.compute_target(self.anchors,
                        np.array(list(map(round, pair['relative_cxcywh']))),
                        pair['other_cxcywhs'])
        pair['reg_target'] = reg_target
        pair['conf_target'] = conf_target 
        pair['score_target'] = score_target 
        pair['hard_neg_weights'] = hard_neg_weights

        if DEBUG:
            img = pair['instance_img']
            size_x = config.instance_img_size
            n = config.anchor_num
            mask = np.where(np.max(conf_target.reshape(n, -1), axis=0).reshape(19,19)>0)
            for i in range(len(mask[0])):
                y, x = int(8*(mask[0][i]-9)+size_x/2), int(size_x/2+8*(mask[1][i]-9))               
                cv2.circle(img, (x, y), 4, (0,0,255), 2)
            mask = np.where(np.max(score_target.reshape(n, -1), axis=0).reshape(19, 19)>0)
            for i in range(len(mask[0])):
                y, x = int(8*(mask[0][i]-9)+size_x/2), int(8*(mask[1][i]-9)+size_x/2)               
                cv2.circle(img, (x, y), 2, (255,255,128), 2)
            boxes = pair['other_cxcywhs']
            
            for b in boxes:
                x1, y1 = int((size_x + 1) / 2 +b[0]- b[2] / 2), int((size_x + 1) / 2 + b[1]- b[3] / 2)
                x2, y2 = int((size_x + 1) / 2 +b[0] + b[2] / 2), int((size_x + 1) / 2 + b[1] + b[3] / 2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 2)

            cv2.imwrite('debug/instance_pos.jpg', img)
        return pair

    def compute_target(self, anchors, box, other_boxes):
        regression_target = self.box_transform(anchors, box)
        base_iou = self.compute_iou(anchors, box)
        other_ious = self.compute_ious(anchors, other_boxes)
        other_iou = np.max(other_ious, axis=1, keepdims=True) # 与其他样本的最大iou
        conf_target = -np.ones(len(base_iou), dtype=np.float32)
        # 与样本的iou>th且与其他样本的iou<th才认为是正样本
        pos_index = np.where(np.logical_and(base_iou > config.pos_threshold, other_iou < config.pos_threshold))[0]
        neg_index = np.where((base_iou < config.neg_threshold))[0]
        conf_target[pos_index] = 1
        conf_target[neg_index] = 0

        score_target = -np.ones(len(base_iou), dtype=np.float32)
        # 与任意目标的iou>th,则认为是行人
        pos_index = np.where(np.logical_or(base_iou > config.pos_threshold, other_iou > config.pos_threshold))[0]
        # pos_index = np.where(other_iou > config.pos_threshold)[0]
        neg_index = np.where(np.logical_and(base_iou < config.neg_threshold,other_iou < config.neg_threshold))[0]
        score_target[pos_index] = 1
        score_target[neg_index] = 0

        hard_neg_weights = np.zeros(len(base_iou), dtype=np.float32)
        neg_index = np.where(np.logical_and(other_iou>config.pos_threshold, base_iou < config.pos_threshold))[0]
        if len(neg_index) > 0: hard_neg_weights[neg_index] = other_iou[neg_index,0]
        return regression_target, conf_target, score_target, hard_neg_weights

    def _transform(self, pair):
        pair['instance_img'] = SV_augmentation(pair['instance_img'])
        pair['exemplar_img'] = SV_augmentation(pair['exemplar_img'])
        return pair
        
    def compute_iou(self, anchors, box):
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(box).ndim == 1:
            box = np.array(box)[None, :]
        else:
            box = np.array(box)
        gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

        anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
        anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
        anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
        anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

        gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
        gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
        gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
        gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

        xx1 = np.max([anchor_x1, gt_x1], axis=0)
        xx2 = np.min([anchor_x2, gt_x2], axis=0)
        yy1 = np.max([anchor_y1, gt_y1], axis=0)
        yy2 = np.min([anchor_y2, gt_y2], axis=0)

        inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                               axis=0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        return iou

    def compute_ious(self, anchors, boxes):
        if len(boxes) == 0:
            return np.zeros((len(anchors),1))
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(boxes).ndim == 1:
            boxes = np.array(boxes)[None, :]
        else:
            boxes = np.array(boxes)

        anchors = anchors[:, np.newaxis]
        gt_boxes = boxes[np.newaxis]
        anchor_x1 = anchors[:, :, 0] - anchors[:, :, 2] / 2 + 0.5
        anchor_x2 = anchors[:, :, 0] + anchors[:, :, 2] / 2 - 0.5
        anchor_y1 = anchors[:, :, 1] - anchors[:, :, 3] / 2 + 0.5
        anchor_y2 = anchors[:, :, 1] + anchors[:, :, 3] / 2 - 0.5

        gt_x1 = gt_boxes[:, :, 0] - gt_boxes[:, :, 2] / 2 + 0.5
        gt_x2 = gt_boxes[:, :, 0] + gt_boxes[:, :, 2] / 2 - 0.5
        gt_y1 = gt_boxes[:, :, 1] - gt_boxes[:,:, 3] / 2 + 0.5
        gt_y2 = gt_boxes[:, :, 1] + gt_boxes[:,:, 3] / 2 - 0.5

        xx1 = np.maximum(anchor_x1, gt_x1)
        xx2 = np.minimum(anchor_x2, gt_x2)
        yy1 = np.maximum(anchor_y1, gt_y1)
        yy2 = np.minimum(anchor_y2, gt_y2)

        inter_area = np.maximum(xx2-xx1, 0) * np.maximum(yy2-yy1, 0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        ious = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        return ious

    def box_transform(self, anchors, gt_box):
        anchor_xctr = anchors[:, :1]
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        gt_cx, gt_cy, gt_w, gt_h = gt_box

        target_x = (gt_cx - anchor_xctr) / anchor_w
        target_y = (gt_cy - anchor_yctr) / anchor_h
        target_w = np.log(gt_w / anchor_w)
        target_h = np.log(gt_h / anchor_h)
        regression_target = np.hstack((target_x, target_y, target_w, target_h))
        return regression_target
    
def collate_fn(batch):
    imgs1, imgs2, reg_targets, cls_targets, scr_targets, hnw_targets = [], [], [], [], [], []
    for pairs in batch:
        imgs1.append(torch.from_numpy(pairs['exemplar_img'].astype(np.float32)).permute(2, 0, 1))
        imgs2.append(torch.from_numpy(pairs['instance_img'].astype(np.float32)).permute(2, 0, 1))
        reg_targets.append(torch.from_numpy(pairs['reg_target']))
        cls_targets.append(torch.from_numpy(pairs['conf_target']).long())
        scr_targets.append(torch.from_numpy(pairs['score_target']).long())
        hnw_targets.append(torch.from_numpy(pairs['hard_neg_weights']))
    return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), torch.stack(reg_targets, dim=0),\
        torch.stack(cls_targets, dim=0), torch.stack(scr_targets, dim=0), torch.stack(hnw_targets, dim=0)             


if __name__ == '__main__':
    # 测试数据加载器的每个方法的功能
    # 1. 初始化
    train_data_roots = ('/media/nlpr/zwzhou/数据资源/MOT16/train', '/media/nlpr/zwzhou/数据资源/MOT15/train')
    train_subsets = (('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-10', 'MOT16-11', 'MOT16-13'),
                    ('ADL-Rundle-6', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
                    'KITTI-17', 'PETS09-S2L1', 'TUD-Stadtmitte', 'Venice-2'))
    mot_dataset = DDDataset(train_data_roots, train_subsets)
    print('There are %d subsets'%len(mot_dataset.img_files))
    print('There are %d label files'%len(mot_dataset))
    print('anchor size: ', mot_dataset.anchors.shape)

    # 2. _get_pairs
    # mot_dataset.get_pairs(img_path1, label_path1, img_path2, label_path2)
    # 3. _target, 创建gt
    # mot_dataset._target()
    # 4. 验证dataloader
    from torch.utils.data import DataLoader
    mot_dataloader = DataLoader(mot_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn)
    for imgs1, imgs2, reg_targets, cls_targets, scr_targets, hnw_targets in mot_dataloader:
        print(imgs1.size())
        print(imgs2.size())
        print(reg_targets.size())
        print(cls_targets.size())
        print(scr_targets.size())
        print(hnw_targets.size())
        break


        

        