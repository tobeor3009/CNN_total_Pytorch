import pickle5 as pickle
import numpy as np
from glob import glob
import os
import shutil
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
from .utils import *
import pydicom


DATASET_VERSION = "20211014"


def plt_auroc(nm_cases, ab_cases, results, png_name):
    for split in ["internal_validation", "external_validation"]:
        "plot results"
        legend = []
        plt.figure(figsize=(14, 14))

        pos_cases = ""
        neg_cases = ""

        total_nm_probs = []
        total_ab_probs = []

        for nm_case in nm_cases:
            total_nm_probs += results[split][nm_case]
            pos_cases += f"{nm_case};"

        for ab_case in ab_cases:
            total_ab_probs += results[split][ab_case]
            neg_cases += f"{ab_case};"

        total_nm_probs = np.array(total_nm_probs)
        total_ab_probs = np.array(total_ab_probs)

        testy = np.concatenate(
            (np.ones(len(total_ab_probs)), np.zeros(len(total_nm_probs))))
        probs = np.concatenate(
            (np.array(total_ab_probs), np.array(total_nm_probs)))

        "draw figures"
        fpr, tpr, thresholds = roc_curve(testy, probs)
        auroc = roc_auc_score(testy, probs)

        plt.plot(fpr, tpr, marker=".",)
        legend.append(f"{pos_cases} vs. {neg_cases} AUROC = {auroc:.2f}")

        plt.plot([0, 1], [0, 1], linestyle='--')

        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")

        plt.legend(legend,
                   fontsize='xx-large',
                   loc='lower right')
        plt.savefig(os.path.join('figures', result_dir,
                    split, metric, 'auroc', png_name))
        plt.close()


def calculate_opt_threshold(nm_cases,
                            ab_cases,
                            anomaly_scores,
                            datasets,
                            split="internal_validation",
                            return_auroc=False,
                            k_fold=100,
                            mode="youden"
                            ):
    total_nm_probs = []
    total_ab_probs = []

    pos_cases = ""
    neg_cases = ""

    for severity in nm_cases:
        for fname in datasets[split]["tuning"][severity]:
            total_nm_probs.append(anomaly_scores[split][fname])
        pos_cases += f"{severity};"

    for severity in ab_cases:
        for fname in datasets[split]["tuning"][severity]:
            total_ab_probs.append(anomaly_scores[split][fname])
        neg_cases += f"{severity};"

    total_ab_probs = np.array(total_ab_probs)
    total_nm_probs = np.array(total_nm_probs)

    aurocs = []
    optimal_thresholds = []
    for k in range(k_fold):
        if k_fold == 1:
            testy = np.concatenate(
                [np.ones(len(total_ab_probs)), np.zeros(len(total_nm_probs))])
            probs = np.concatenate([total_ab_probs, total_nm_probs])

        elif k_fold > 1:
            # random 함수 바꿔야함
            ab_idxes = np.unique(np.random.randint(
                0, len(total_ab_probs), int(0.8 * len(total_ab_probs))))
            nm_idxes = np.unique(np.random.randint(
                0, len(total_nm_probs), int(0.8 * len(total_nm_probs))))
            if len(ab_idxes) == 0 or len(nm_idxes) == 0:
                continue

            kfold_ab_probs = total_ab_probs[ab_idxes]
            kfold_nm_probs = total_nm_probs[nm_idxes]

            testy = np.concatenate(
                [np.ones_like(kfold_ab_probs), np.zeros_like(kfold_nm_probs)])
            probs = np.concatenate([kfold_ab_probs, kfold_nm_probs])

        fpr, tpr, thresholds = roc_curve(testy, probs)
        aurocs.append(roc_auc_score(testy, probs))

        # calculate opt threshold
        if mode == "youden":
            optimal_idx = np.argmax(np.abs(tpr - fpr))
            optimal_thresholds.append(thresholds[optimal_idx])
        if mode == "0.95sensitivity":
            optimal_thresholds.append(thresholds[tpr > 0.945][0])  # .min()
        if mode == "1.00sensitivity":
            optimal_thresholds.append(thresholds[tpr > 0.995][0])  # .min()

    auroc = np.array(aurocs).mean()
    optimal_threshold = np.array(optimal_thresholds).mean()

    if return_auroc:
        return optimal_threshold, auroc
    else:
        return optimal_threshold


def get_confidence_interval(error, n, significance=0.95):
    """
    https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    corresponding significance level are as follows:
        - 1.64 (90%)
        - 1.96 (95%)
        - 2.33 (98%)
        - 2.58 (99%)
    """
    significance_level = {
        0.90: 1.64,
        0.95: 1.96,
        0.98: 2.33,
        0.99: 2.58,
    }

    # Q_1 = AUC / (2 - AUC)
    # Q_2 = 2 * (AUC ** 2) / (1 + AUC)

    return error


def get_performance(y_true, y_pred, thres, n_bootstraps=10000, alpha=0.95):
    bootstrapped_scores = {
        "sens": [],
        "spec": [],
        "acc": [],
    }
#     rng = np.random.RandomState(rng_seed)

    i = 0
    while i < n_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        #         indices = rng.randint(0, len(y_pred), len(y_pred))
        indices = np.unique(np.random.randint(0, len(y_pred), len(y_pred)))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        i += 1

        p = np.sum(y_true[indices])
        n = np.sum((1 - y_true[indices]))

        tp = np.sum((y_pred[indices] >= thres) * y_true[indices])
        fp = np.sum((y_pred[indices] >= thres) * (1 - y_true[indices]))
        tn = np.sum((y_pred[indices] < thres) * (1 - y_true[indices]))
        fn = np.sum((y_pred[indices] < thres) * y_true[indices])

        # print(f"{p}p, {n}n, {tp}tp , {fp}fp, {tn}tn, {fn}fn")
        assert p == tp + fn, print(p, tp + fn)
        assert n == tn + fp, print(n, tn + fp)

        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        acc = (tp + tn) / (tp + tn + fn + fp)

        bootstrapped_scores["sens"].append(sens)
        bootstrapped_scores["spec"].append(spec)
        bootstrapped_scores["acc"].append(acc)

    # confidence intervals
    CI_scores = {}
    for label, scores in bootstrapped_scores.items():
        scores.sort()
        p = [((1.0 - alpha) / 2.0) * 100, (alpha + ((1.0 - alpha) / 2.0)) * 100]
        lower, upper = np.percentile(scores, p)
        CI_scores[label] = np.array(
            [score for score in scores if lower <= score <= upper])

    return CI_scores


def get_roc(y_true, y_pred, n_bootstraps=10000, alpha=0.95):
    bootstrapped_scores = []

    i = 0
    while i < n_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        #         indices = rng.randint(0, len(y_pred), len(y_pred))
        indices = np.unique(np.random.randint(0, len(y_pred), len(y_pred)))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        i += 1
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    bootstrapped_scores.sort()

    # confidence intervals
    p = [((1.0 - alpha) / 2.0) * 100, (alpha + ((1.0 - alpha) / 2.0)) * 100]
    lower, upper = np.percentile(bootstrapped_scores, p)
    scores_in_ci = [
        score for score in bootstrapped_scores if lower <= score <= upper]
    return scores_in_ci


def get_results(result_path, normal_stat, metric, step=-1, normalize=True, splits=["internal_validation", "external_validation"]):
    patients = {
        "internal_validation": glob(f"../Dataset/Asan_Brain_CT/test/internal_validation/{DATASET_VERSION}/*/*"),
        "external_validation": glob(f"../Dataset/Asan_Brain_CT/test/external_validation/{DATASET_VERSION}/*/*"),
    }

    severities = ['Immediate', 'Urgent', 'Indeterminate', 'Benign', 'Normal']
    results = {split: {} for split in splits}
    for split in splits:
        for severity in severities:
            dirs = sorted(
                glob(os.path.join(result_path, split, severity, "*")))
            for dir in (dirs):
                patient_id = dir.split('/')[-1]
#                 for path in patients[split]:
#                     if patient_id in path:
#                         dcms = glob(os.path.join(path, "2", "*.dcm"))
#                 pixel_spacing = pydicom.dcmread(dcms[0]).PixelSpacing[0]

                fname = dir.split('/')[-1]
                history_pkl = os.path.join(dir, "history", "history.pkl")
                if os.path.exists(history_pkl):
                    history = load_obj(history_pkl)
                    diff_score = history[metric][step]

                    score = []
                    for i in range(len(diff_score)):
                        #                         diff_score[i] /= pixel_spacing ** 2
                        if normalize:
                            mean, std = normal_stat[i]
                            normalized_score = (
                                diff_score[i] - mean) / (std + 1e-5)
                            score.append(normalized_score)
                        else:
                            score.append(diff_score[i])

                    score_np = np.array(score)
                    results[split][fname] = score_np.sum()
    return results


def get_normal_stat(result_path, metric, step=-1):
    patients = {
        "internal_validation": glob(f"../Dataset/Asan_BrainCT/test/internal_validation/{DATASET_VERSION}/*/*"),
        "external_validation": glob(f"../Dataset/Asan_BrainCT/test/external_validation/{DATASET_VERSION}/*/*"),
    }

    normal_stat = {}
    for split in ["internal_validation", "external_validation"]:
        dirs = sorted(glob(os.path.join(result_path, split, "Normal*", "*")))

        for dir in (dirs):
            patient_id = dir.split('/')[-1]
#             for path in patients[split]:
#                 if patient_id in path:
#                     dcms = glob(os.path.join(path, "2", "*.dcm"))
#             pixel_spacing = 1 #pydicom.dcmread(dcms[0]).PixelSpacing[0]

            history_pkl = os.path.join(dir, "history", "history.pkl")
            if os.path.exists(history_pkl):
                history = load_obj(history_pkl)
                diff_score = history[metric][step]
                for i, score in enumerate(diff_score):
                    #                     score /= (pixel_spacing ** 2)
                    if i in normal_stat:
                        normal_stat[i].append(score)
                    else:
                        normal_stat[i] = [score]
    for k, v in normal_stat.items():
        normal_stat[k] = [np.array(v).mean(), np.array(v).std()]
    return normal_stat


def plot_confusion_matrix(cm,
                          target_names,
                          save_path,
                          title='Confusion matrix',
                          cmap=None,
                          figsize=(8, 6)):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    augments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    #  https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    tn = cm[0, 0]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]

    cm = np.array([[tp, fn],
                   [fp, tn]])

    tn = cm_normalized[0, 0]
    fn = cm_normalized[1, 0]
    tp = cm_normalized[1, 1]
    fp = cm_normalized[0, 1]

    cm_normalized = np.array([[tp, fn],
                              [fp, tn]])

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0, fontsize=12)
        plt.yticks(tick_marks, target_names, rotation=90, fontsize=12)

    thresh = cm_normalized.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm_normalized[j, i]:.2f}\n({cm[j, i]})",
                 horizontalalignment="center",
                 color="white" if cm_normalized[j, i] > thresh else "black",
                 fontsize=14)

    # plt.title(title, fontsize=20)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_path)
    plt.close()
