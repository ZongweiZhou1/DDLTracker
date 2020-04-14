import numpy as np 

class Config(object):
    exemplar_img_size = 127
    instance_img_size = 271
    
    epoches = 100
    initial_lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    lr_decay = 0.8685
    r_pos = 16
    r_neg = 48
    clip = 100
    train_batch_size = 32

    anchor_scales = np.array([6, 8, 10, 12])
    anchor_ratios = np.array([2.4])
    anchor_num = len(anchor_scales) * len(anchor_ratios) # 5
    score_size = int((instance_img_size-exemplar_img_size)//8)+1
    total_anchor_num = score_size**2 * anchor_num

    stride = 8

    pos_threshold = 0.5
    neg_threshold = 0.3

    context = 0.5 
    penalty_k = 0.055
    window_influence = 0.42 
    eps = 0.01 
    scale_lr = 0.295

    max_translate = 12
    scale_resize = 0.15

    alpha = 0.4
    beta = 0.4
    lam_det = 1
    lam_reg = 5

config = Config()