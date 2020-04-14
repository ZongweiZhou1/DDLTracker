# --*-- coding:utf-8 --*--
import torch 
import torch.nn.functional as F 
import random 
import numpy as np 


def rpn_cross_entropy(pred, target, num_pos, num_neg, weights=None, alpha=0.4):
    """ 计算rpn 的分类损失,判别模板的相似度
    :param pred:        rpn cls 分支的输出, B x -1 x 2
    :param target:      rpn_cls 分支的gt, B x -1 x 1,表示每个anchor是否是正样本
    :param num_pos:     一个sample中正样本的个数上限
    :param num_neg:     一个sample中负样本的个数上限
    :param weights:     为了增强鉴别性,对于难分样本增加较大权重惩罚, weights可以使用anchor对应行人的置信度的target, B x -1 x 2
    :param alpha:       distractor-aware loss的权重
    """
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
        min_neg = int(min(len(np.where(target[batch_id].cpu() == 0)[0]), num_neg))
        pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
        neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist() 

        pos_index_random = random.sample(pos_index, min_pos)
        if len(pos_index) > 0:
            pos_loss_bid_final = F.cross_entropy(pred[batch_id][pos_index_random],
                            target[batch_id][pos_index_random], reduction='none')
            neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), min_neg)
            neg_loss_bid_final = F.cross_entropy(pred[batch_id][neg_index_random],
                            target[batch_id][neg_index_random], reduction='none')
        else:
            pos_loss_bid_final = torch.FloatTensor([0]).cuda()
            neg_index_random = random.sample(neg_index, min_neg)
            neg_loss_bid_final = F.cross_entropy(pred[batch_id][neg_index_random],
                                                target[batch_id][neg_index_random], reduction='none')
        hard_neg_bid_final = torch.FloatTensor([0]).mean().to(target.device)
        if weights is not None:
            hard_neg_mask = weights[batch_id]>0.1
            if hard_neg_mask.sum() > 0:
                hard_neg_loss = F.cross_entropy(pred[batch_id][hard_neg_mask], 
                                                target[batch_id].new_zeros(hard_neg_mask.sum()), reduction='none')
                hard_neg_bid_final = (hard_neg_loss*weights[batch_id][hard_neg_mask]).mean()
        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2 + alpha * hard_neg_bid_final

        loss_all.append(loss_bid)
    finall_loss = torch.stack(loss_all).mean()
    return finall_loss

def get_csc_loss(anchors):
    """ anchors: N x 4, ndarray, cx, cy, w,h"""
    gt_box = torch.mean(anchors, dim=0, keepdim=True)
    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2

    xx1 = torch.max(anchor_x1, gt_x1)
    xx2 = torch.min(anchor_x2, gt_x2)
    yy1 = torch.max(anchor_y1, gt_y1)
    yy2 = torch.min(anchor_y2, gt_y2)

    inter_area = (xx2-xx1).clamp(min=1) * (yy2-yy1).clamp(min=1)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return 1-iou.mean()

def rpn_smoothL1(output, target, label, anchors, num_pos=16, beta=0.4):
    loss_all, loss_bid_a, loss_csc_a = [], [], []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos)
        pos_index = np.where(label[batch_id].cpu() == 1)[0]
        pos_index = random.sample(pos_index.tolist(), min_pos)
        if len(pos_index) > 0:
            loss_reg = F.smooth_l1_loss(output[batch_id][pos_index], target[batch_id][pos_index])
            # 还需计算regression boxes的一致性
            offsets = output[batch_id]  # anchor_total_num x 4
            pos_offsets, pos_anchors = offsets[pos_index], anchors[pos_index]
            pos_offsets[:, 0] = pos_offsets[:, 0] * pos_anchors[:, 2] + pos_anchors[:, 0]
            pos_offsets[:, 1] = pos_offsets[:, 1] * pos_anchors[:, 3] + pos_anchors[:, 1]
            # pos_offsets[:, 2] = torch.exp(torch.clamp_max_(pos_offsets[:, 2], 5)) * pos_anchors[:, 2]
            # pos_offsets[:, 3] = torch.exp(torch.clamp_max_(pos_offsets[:, 3], 5)) * pos_anchors[:, 3]
            pos_offsets[:, 2] = torch.exp(pos_offsets[:, 2]) * pos_anchors[:, 2]
            pos_offsets[:, 3] = torch.exp(pos_offsets[:, 3]) * pos_anchors[:, 3]

            loss_csc = get_csc_loss(pos_offsets)
            loss_bid = loss_reg + beta * loss_csc
            loss_bid_a.append(loss_reg)
            loss_csc_a.append(loss_csc)
        else:
            loss_bid = torch.FloatTensor([0]).cuda()[0]
        loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss

