import torch
import torch.nn.functional as F
from functools import partial
from .util import SMOOTH
from .util import get_clip, get_rev, get_one_hot_argmax
# import for compatibility with previous code
from .class_loss import get_bce_loss_class, get_cce_loss_class, get_focal_loss_class

def get_seg_dim(tensor):
    num_dims = len(tensor.shape)
    axis = tuple(range(2, num_dims))
    return axis

def get_bce_loss(y_pred, y_true, per_image=False, smooth=SMOOTH):
    axis = get_seg_dim(y_true)
    y_pred, y_true, y_true_rev, y_pred_rev = get_clip(y_pred, y_true, smooth)
    gt_negative_term = -(y_true_rev) * torch.log(y_pred_rev)
    gt_positive_term = -y_true * torch.log(y_pred)
    per_image_loss = torch.mean(gt_positive_term + gt_negative_term, dim=axis)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)

def get_focal_loss(y_pred, y_true, per_image=False, gamma=2.0, smooth=SMOOTH):
    axis = get_seg_dim(y_true)
    y_pred, y_true, y_true_rev, y_pred_rev = get_clip(y_pred, y_true, smooth)
    gt_negative_term = -(y_pred ** gamma) * (y_true_rev) * torch.log(y_pred_rev)
    gt_positive_term = -(y_pred_rev ** gamma) * y_true * torch.log(y_pred)
    per_image_loss = torch.mean(gt_positive_term + gt_negative_term, dim=axis)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)

def get_dice_loss(y_pred, y_true, log=False, per_image=False, smooth=SMOOTH):
    axis = get_seg_dim(y_true)
    tp = torch.sum(y_true * y_pred, axis=axis)
    fp = torch.sum(y_pred, axis=axis) - tp
    fn = torch.sum(y_true, axis=axis) - tp
    dice_score_per_image = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    if log:
        dice_score_per_image = -1 * torch.log(dice_score_per_image)
    else:
        dice_score_per_image = 1 - dice_score_per_image

    if per_image:
        return dice_score_per_image
    else:
        return torch.mean(dice_score_per_image)


def get_tversky_loss(y_pred, y_true, beta=0.7,
                     log=False, per_image=False, smooth=SMOOTH):
    axis = get_seg_dim(y_true)
    alpha = 1 - beta
    tp = torch.sum(y_true * y_pred, axis=axis)
    fp = torch.sum(y_pred, axis=axis) - tp
    fn = torch.sum(y_true, axis=axis) - tp
    dice_score_per_image = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    if log:
        dice_score_per_image = -1 * torch.log(dice_score_per_image)
    else:
        dice_score_per_image = 1 - dice_score_per_image

    if per_image:
        return dice_score_per_image
    else:
        return torch.mean(dice_score_per_image)

def get_propotional_loss(y_pred, y_true, beta=0.7,
                         log=False, per_image=False, smooth=SMOOTH):
    
    y_pred_rev, y_true_rev = get_rev(y_pred, y_true)
    num_dims = len(y_true.shape)
    axis = tuple(range(2, num_dims))

    alpha = 1 - beta
    prevalence = torch.mean(y_true, dim=axis)
    prevalence = 2 * torch.sigmoid(1e3 * prevalence) - 1
    negative_ratio = 1 - prevalence
    positive_ratio = prevalence

    tp = torch.sum(y_true * y_pred, dim=axis)
    tn = torch.sum(y_true_rev * y_pred_rev, dim=axis)
    fp = torch.sum(y_pred, dim=axis) - tp
    fn = torch.sum(y_true, dim=axis) - tp
    
    negative_score = (tn + smooth) / (tn + beta * fn + alpha * fp + smooth)
    positive_score = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    negative_score_propo = negative_score * negative_ratio
    positive_score_propo = positive_score * positive_ratio
    score_per_image = negative_score_propo + positive_score_propo

    if log:
        score_per_image = -1 * torch.log(score_per_image)
    else:
        score_per_image = 1 - score_per_image

    if per_image:
        return score_per_image
    else:
        return torch.mean(score_per_image)


def get_dice_bce_loss(y_pred, y_true, log=False, per_image=False):
    dice_loss = get_dice_loss(y_pred, y_true, log=log, per_image=per_image)
    bce_loss = get_bce_loss(y_pred, y_true)
    return (dice_loss + bce_loss) / 2

def get_tversky_bce_loss(y_pred, y_true, log=False, per_image=False):
    tversky_loss = get_tversky_loss(y_pred, y_true, log=log, per_image=per_image)
    bce_loss = get_bce_loss(y_pred, y_true)
    return (tversky_loss + bce_loss) / 2

def get_propotional_bce_loss(y_pred, y_true, log=False, per_image=False):
    propotional_loss = get_propotional_loss(y_pred, y_true, log=log, per_image=per_image)
    bce_loss = get_bce_loss(y_pred, y_true)
    return (propotional_loss + bce_loss) / 2

def get_dice_focal_loss(y_pred, y_true, log=False, per_image=False):
    dice_loss = get_dice_loss(y_pred, y_true, log=log, per_image=per_image)
    focal_loss = get_focal_loss(y_pred, y_true)
    return (dice_loss + focal_loss) / 2

def get_tversky_focal_loss(y_pred, y_true, log=False, per_image=False):
    tversky_loss = get_tversky_loss(y_pred, y_true, log=log, per_image=per_image)
    focal_loss = get_focal_loss(y_pred, y_true)
    return (tversky_loss + focal_loss) / 2

def get_propotional_focal_loss(y_pred, y_true, log=False, per_image=False):
    propotional_loss = get_propotional_loss(y_pred, y_true, log=log, per_image=per_image)
    focal_loss = get_focal_loss(y_pred, y_true)
    return (propotional_loss + focal_loss) / 2

def get_dice_bce_focal_loss(y_pred, y_true, log=False, per_image=False):
    dice_loss = get_dice_loss(y_pred, y_true, log=log, per_image=per_image)
    bce_loss = get_bce_loss(y_pred, y_true)
    focal_loss = get_focal_loss(y_pred, y_true)
    return (dice_loss + bce_loss + focal_loss) / 3

def get_tversky_bce_focal_loss(y_pred, y_true, log=False, per_image=False):
    tversky_loss = get_tversky_loss(y_pred, y_true, log=log, per_image=per_image)
    bce_loss = get_bce_loss(y_pred, y_true)
    focal_loss = get_focal_loss(y_pred, y_true)
    return (tversky_loss + bce_loss + focal_loss) / 3

def get_propotional_bce_focal_loss(y_pred, y_true, log=False, per_image=False):
    propotional_loss = get_propotional_loss(y_pred, y_true, log=log, per_image=per_image)
    bce_loss = get_bce_loss(y_pred, y_true)
    focal_loss = get_focal_loss(y_pred, y_true)
    return (propotional_loss + bce_loss + focal_loss) / 3

def final_loss_fn(y_pred, y_true, region_loss, log=False, per_image=False, exclude_background=True):
    C = y_true.shape[1]
    if C > 1 and exclude_background:
        y_pred = y_pred[:, 1:]
        y_true = y_true[:, 1:]
    return region_loss(y_pred, y_true, log=log, per_image=per_image)

def get_loss_fn(loss_select, per_image=False, exclude_background=True):
    if loss_select == "dice":
        region_loss = get_dice_loss
    elif loss_select == "tversky":
        region_loss = get_tversky_loss
    elif loss_select == "propotional":
        region_loss = get_propotional_loss
    elif loss_select == "dice_bce":
        region_loss = get_dice_bce_loss
    elif loss_select == "tversky_bce":
        region_loss = get_tversky_bce_loss
    elif loss_select == "propotional_bce":
        region_loss = get_propotional_bce_loss
    elif loss_select == "dice_focal":
        region_loss = get_dice_focal_loss
    elif loss_select == "tversky_focal":
        region_loss = get_tversky_focal_loss
    elif loss_select == "propotional_focal":
        region_loss = get_propotional_focal_loss
    elif loss_select == "dice_bce_focal":
        region_loss = get_dice_bce_focal_loss
    elif loss_select == "tversky_bce_focal":
        region_loss = get_tversky_bce_focal_loss
    elif loss_select == "propotional_bce_focal":
        region_loss = get_propotional_bce_focal_loss
    else:
        region_loss = get_dice_loss
        print("region loss selected as dice loss")
           
    return partial(final_loss_fn, region_loss=region_loss, per_image=per_image, exclude_background=exclude_background)

def get_recon_loss_follow_seg(y_recon_pred, y_recon_gt, y_seg_pred, per_image=False):
    recon_image_channel = y_recon_pred.size(1)
    y_seg_pred_weight = 2 * torch.sigmoid(25 * y_seg_pred[:, 1]) - 1
    y_seg_pred_weight = y_seg_pred_weight.unsqueeze(1).repeat(1, recon_image_channel, 1, 1, 1)
    recon_loss = torch.abs(y_recon_pred - y_recon_gt) * y_seg_pred_weight
    
    if per_image:
        recon_loss = torch.mean(recon_loss)
    else:
        target_mean_dim_tuple = range(1, recon_loss)
        recon_loss = torch.mean(target_mean_dim_tuple, dim=target_mean_dim_tuple, keepdim=False) 
    return per_image

def get_thresholded_seg_mask(y_pred, y_true, mask_threshold=0.5):
    seg_channel = y_true.size(1)
    if seg_channel > 1:
        y_pred = y_pred[:, 1:]
        y_true = y_true[:, 1:]
    if mask_threshold is not None:
        y_pred = (y_pred >= mask_threshold).float()
    else:
        y_pred = get_one_hot_argmax(y_pred)
    return y_pred, y_true

@torch.no_grad
def get_dice_score(y_pred, y_true, mask_threshold=0.5, per_image=False, per_channel=False):
    y_pred, y_true = get_thresholded_seg_mask(y_pred, y_true, mask_threshold=mask_threshold)
    dice_loss = get_dice_loss(y_pred, y_true, log=False, per_image=per_image)
    dice_score = 1 - dice_loss
    if (not per_channel) and per_image:
        dice_score = dice_score.mean(1)
    return dice_score

@torch.no_grad
def get_seg_precision(y_pred, y_true, mask_threshold=0.5, per_image=False, per_channel=False, smooth=SMOOTH):
    y_pred, y_true = get_thresholded_seg_mask(y_pred, y_true, mask_threshold=mask_threshold)
    num_dims = y_true.ndim
    axis = tuple(range(1 + int(per_channel), num_dims))
    tp = torch.sum(y_true * y_pred, dim=axis)
    fp = torch.sum(y_pred, dim=axis) - tp
    precision = (tp + smooth) / (tp + fp + smooth)
    if per_image:
        return precision
    else:
        return torch.mean(precision)

@torch.no_grad
def get_seg_recall(y_pred, y_true, mask_threshold=0.5, per_image=False, per_channel=False, smooth=SMOOTH):
    y_pred, y_true = get_thresholded_seg_mask(y_pred, y_true, mask_threshold=mask_threshold)
    num_dims = y_true.ndim
    axis = tuple(range(1 + int(per_channel), num_dims))
    tp = torch.sum(y_true * y_pred, dim=axis)
    fn = torch.sum(y_true, dim=axis) - tp
    recall = (tp + smooth) / (tp + fn + smooth)
    if per_image:
        return recall
    else:
        return torch.mean(recall)

@torch.no_grad    
def accuracy_metric(preds, gt, per_image=False):
    # 예측된 클래스 얻기 (가장 높은 확률을 가진 클래스)
    predicted_classes = torch.argmax(preds, dim=1)
    # Ground truth 클래스 얻기
    true_classes = torch.argmax(gt, dim=1)
    # 정확하게 분류된 샘플 수 계산
    corrects = (predicted_classes == true_classes).float()
    
    if per_image:
        accuracy = corrects
    else:
        accuracy = corrects.mean()

    return accuracy