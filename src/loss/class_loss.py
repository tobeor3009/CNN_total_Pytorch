from functools import partial
import torch
from .util import SMOOTH
from .util import get_clip, get_rev, get_one_hot_argmax

def get_bce_loss_class(y_pred, y_true, per_image=False, smooth=SMOOTH):
    axis = 1
    y_pred, y_true, y_true_rev, y_pred_rev = get_clip(y_pred, y_true, smooth)
    gt_negative_term = -(y_true_rev) * torch.log(y_pred_rev)
    gt_positive_term = -y_true * torch.log(y_pred)
    per_image_loss = torch.mean(gt_positive_term + gt_negative_term, dim=axis)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)
    
def get_cce_loss_class(y_pred, y_true, per_image=False, smooth=SMOOTH):
    axis = 1
    y_pred, y_true, _, _ = get_clip(y_pred, y_true, smooth)
    y_pred = torch.log(y_pred)
    per_image_loss = -torch.sum(y_true * y_pred, dim=axis)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)
    
def get_focal_loss_class(y_pred, y_true, per_image=False, gamma=2.0, smooth=SMOOTH):
    axis = 1
    y_pred, y_true, y_true_rev, y_pred_rev = get_clip(y_pred, y_true, smooth)
    gt_negative_term = -(y_pred ** gamma) * (y_true_rev) * torch.log(y_pred_rev)
    gt_positive_term = -(y_pred_rev ** gamma) * y_true * torch.log(y_pred)
    per_image_loss = torch.mean(gt_positive_term + gt_negative_term, dim=axis)
    if per_image:
        return per_image_loss
    else:
        return torch.mean(per_image_loss)

def get_loss_fn(loss_select, per_image=False):
    if loss_select in ["bce", "binary_cross_entropy"]:
        class_loss = get_bce_loss_class
    elif loss_select in ["cce", "categorical_cross_entropy"]:
        class_loss = get_cce_loss_class
    elif loss_select == "focal":
        class_loss = get_focal_loss_class
    else:
        class_loss = get_bce_loss_class
        print("class loss selected as bce loss")
    return partial(class_loss, per_image)

def get_thresholded_class_mask(y_pred, mask_threshold=0.5):
    if mask_threshold is not None:
        y_pred = (y_pred >= mask_threshold).float()
    else:
        y_pred = get_one_hot_argmax(y_pred)
    return y_pred

def preprocess_class_metric_data(y_label_pred, y_label, mask_threshold=0.5, per_channel=False, exclude_class_0=False):
    if per_channel and exclude_class_0:
        y_label_pred = y_label_pred[:, 1:]
        y_label = y_label[:, 1:]
    y_label_pred = get_thresholded_class_mask(y_label_pred, mask_threshold)
    
    return y_label_pred, y_label
    
def get_class_accuracy(y_label_pred, y_label, mask_threshold=0.5, per_channel=False, exclude_class_0=False, smooth=SMOOTH):

    y_label_pred, y_label = preprocess_class_metric_data(y_label_pred, y_label, mask_threshold=mask_threshold, 
                                                         per_channel=per_channel, exclude_class_0=exclude_class_0)
    y_label_pred_rev, y_label_rev = get_rev(y_label_pred, y_label)
    target_sum_dim = 0 if per_channel else None
    
    tp = torch.sum(y_label_pred * y_label, dim=target_sum_dim)
    tn = torch.sum(y_label_pred_rev * y_label_rev, dim=target_sum_dim)
    fp = torch.sum(y_label_pred, dim=target_sum_dim) - tp
    fn = torch.sum(y_label, dim=target_sum_dim) - tp
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn)
    return accuracy

def get_class_precison(y_label_pred, y_label, mask_threshold=0.5, per_channel=False, exclude_class_0=False, smooth=SMOOTH):
    y_label_pred, y_label = preprocess_class_metric_data(y_label_pred, y_label, mask_threshold=mask_threshold,
                                                         per_channel=per_channel, exclude_class_0=exclude_class_0)
    tp = torch.sum(y_label_pred * y_label)
    fp = torch.sum(y_label_pred) - tp
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision

def get_class_recall(y_label_pred, y_label, mask_threshold=0.5, per_channel=False, exclude_class_0=False, smooth=SMOOTH):
    y_label_pred, y_label = preprocess_class_metric_data(y_label_pred, y_label, mask_threshold=mask_threshold, 
                                                         per_channel=per_channel, exclude_class_0=exclude_class_0)
    tp = torch.sum(y_label_pred * y_label)
    fn = torch.sum(y_label) - tp
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

def get_class_accuracy_multiclass(y_label_pred, y_label, mask_threshold=0.5, smooth=1e-5, exclude_class_0=True):
    # y_label_pred: B x C, softmax probabilities
    # y_label: B x C, one-hot encoded ground truth
    
    # Threshold the predictions to get binary outputs
    y_label_pred = (y_label_pred >= mask_threshold).float()
    y_label = y_label.float()

    accuracies = []

    # Iterate through each class, starting from class 1 if exclude_class_0 is True
    start_class = 1 if exclude_class_0 else 0
    for class_idx in range(start_class, y_label.shape[1]):
        y_label_pred_class = y_label_pred[:, class_idx]
        y_label_class = y_label[:, class_idx]

        # Calculate true positives, true negatives, false positives, and false negatives
        tp = torch.sum(y_label_pred_class * y_label_class)
        tn = torch.sum((1 - y_label_pred_class) * (1 - y_label_class))
        fp = torch.sum(y_label_pred_class) - tp
        fn = torch.sum(y_label_class) - tp

        # Compute accuracy for the current class
        accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
        accuracies.append(accuracy.item())

    return accuracies