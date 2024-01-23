import torch
import torch.optim.lr_scheduler as lr_scheduler
from functools import partial
AXIS = [2, 3]
SMOOTH = 1e-7


def get_bce_loss(y_pred, y_true, per_image=False, smooth=SMOOTH):
    y_true = torch.clamp(y_true, min=smooth)
    y_pred = torch.clamp(y_pred, min=smooth)
    y_true_reverse = torch.clamp(1 - y_true, min=smooth)
    y_pred_reverse = torch.clamp(1 - y_pred, min=smooth)
    gt_negative_term = -(y_true_reverse) * torch.log(y_pred_reverse)
    gt_positive_term = -y_true * torch.log(y_pred)
    per_image_loss = torch.mean(gt_positive_term + gt_negative_term, dim=AXIS)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)


def get_focal_loss(y_pred, y_true, per_image=False, gamma=2.0, smooth=SMOOTH):
    batch_size = y_pred.size(0)
    y_true = torch.clamp(y_true, min=smooth)
    y_pred = torch.clamp(y_pred, min=smooth)
    y_true_reverse = torch.clamp(1 - y_true, min=smooth)
    y_pred_reverse = torch.clamp(1 - y_pred, min=smooth)
    gt_negative_term = -(y_pred ** gamma) * \
        (y_true_reverse) * torch.log(y_pred_reverse)
    gt_positive_term = -(y_pred_reverse ** gamma) * y_true * torch.log(y_pred)
    per_image_loss = torch.mean(gt_positive_term + gt_negative_term, dim=AXIS)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)


def get_dice_loss(y_pred, y_true, axis=AXIS, log=False, per_image=False, smooth=SMOOTH):
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


def get_tversky_loss(y_pred, y_true, beta=0.7, axis=AXIS, log=False, per_image=False, smooth=SMOOTH):
    alpha = 1 - beta
    tp = torch.sum(y_true * y_pred, axis=axis)
    fp = torch.sum(y_pred, axis=axis) - tp
    fn = torch.sum(y_true, axis=axis) - tp
    dice_score_per_image = (tp + smooth) / \
        (tp + alpha * fp + beta * fn + smooth)
    if log:
        dice_score_per_image = -1 * torch.log(dice_score_per_image)
    else:
        dice_score_per_image = 1 - dice_score_per_image

    if per_image:
        return dice_score_per_image
    else:
        return torch.mean(dice_score_per_image)


def get_propotional_loss(y_pred, y_true, axis=AXIS, log=False, per_image=False, smooth=SMOOTH, beta=0.9):

    alpha = 1 - beta
    prevalence = torch.mean(y_true, dim=axis)

    tp = torch.sum(y_true * y_pred, dim=axis)
    tn = torch.sum((1 - y_true) * (1 - y_pred), dim=axis)
    fp = torch.sum(y_pred, dim=axis) - tp
    fn = torch.sum(y_true, dim=axis) - tp
    negative_score = (tn + smooth) \
        / (tn + beta * fn + alpha * fp + smooth) * (smooth + 1 - prevalence)
    positive_score = (tp + smooth) \
        / (tp + alpha * fn + beta * fp + smooth) * (smooth + prevalence)
    score_per_image = negative_score + positive_score

    if log:
        score_per_image = -1 * torch.log(score_per_image)
    else:
        score_per_image = 1 - score_per_image

    if per_image:
        return score_per_image
    else:
        return torch.mean(score_per_image)


def get_dice_bce_loss(y_pred, y_true, axis=AXIS):
    return get_bce_loss(y_pred, y_true) + get_dice_loss(y_pred, y_true, axis=axis)

def get_tversky_bce_loss(y_pred, y_true, axis=AXIS):
    return get_bce_loss(y_pred, y_true) + get_tversky_loss(y_pred, y_true, axis=axis)


def get_propotional_bce_loss(y_pred, y_true, axis=AXIS):
    return get_bce_loss(y_pred, y_true) + get_propotional_loss(y_pred, y_true, axis=axis)

def get_dice_focal_loss(y_pred, y_true, axis=AXIS):
    return get_focal_loss(y_pred, y_true) + get_dice_loss(y_pred, y_true, axis=axis)

def get_tversky_focal_loss(y_pred, y_true, axis=AXIS):
    return get_focal_loss(y_pred, y_true) + get_tversky_loss(y_pred, y_true, axis=axis)


def get_propotional_focal_loss(y_pred, y_true, axis=AXIS):
    return get_focal_loss(y_pred, y_true) + get_propotional_loss(y_pred, y_true, axis=axis)


def get_loss_fn(loss_select, axis=AXIS):
    if loss_select == "dice":
        region_loss = partial(get_dice_loss, axis=axis)
    elif loss_select == "tversky":
        region_loss = partial(get_tversky_loss, axis=axis)
    elif loss_select == "propotional":
        region_loss = partial(get_propotional_loss, axis=axis)
    elif loss_select == "dice_bce":
        region_loss = partial(get_dice_bce_loss, axis=axis)
    elif loss_select == "tversky_bce":
        region_loss = partial(get_tversky_bce_loss, axis=axis)
    elif loss_select == "propotional_bce":
        region_loss = partial(get_propotional_bce_loss, axis=axis)
    elif loss_select == "dice_focal":
        region_loss = partial(get_dice_focal_loss, axis=axis)
    elif loss_select == "tversky_focal":
        region_loss = partial(get_tversky_focal_loss, axis=axis)
    elif loss_select == "propotional_focal":
        region_loss = partial(get_propotional_focal_loss, axis=axis)

    def final_loss_fn(y_pred, y_true):
        C = y_true.shape[1]
        if C > 1:
            y_pred = y_pred[:, 1:]
            y_true = y_true[:, 1:]
        return region_loss(y_pred, y_true)
#     pointwise_loss = get_bce_loss(y_pred, y_true, per_image=True)
    return final_loss_fn


def get_dice_score(y_pred, y_true, axis=AXIS, mask_threshold=0.5):
    y_pred = (y_pred >= mask_threshold).float()
    y_true = (y_true >= mask_threshold).float()
    dice_loss = get_dice_loss(y_pred, y_true, axis=axis, log=False, per_image=False)
    dice_score = 1 - dice_loss
    return dice_score
