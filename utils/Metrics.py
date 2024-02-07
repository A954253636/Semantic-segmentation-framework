import numpy as np
import config
import six
def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)
    """
    类别数
    """
    n_class = config.class_num
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()  # 拉平预测
        gt_label = gt_label.flatten()  # 拉平标签
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2) \
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou
    # 返回IOU
    # 如果不取背景请加入return iou[:-1]   -1代表最后一个位置的


def calc_semantic_segmentation_f1(confusion):
    precision = np.diag(confusion) / (confusion.sum(axis=1))
    recall = np.diag(confusion) / (confusion.sum(axis=0))
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calc_semantic_segmentation_precision(confusion):
    precision = np.diag(confusion) / (confusion.sum(axis=1))
    return precision


def calc_semantic_segmentation_recall(confusion):
    recall = np.diag(confusion) / (confusion.sum(axis=0))
    return recall


def calc_semantic_segmentation_kappa(confusion):
    observed_agreement = np.trace(confusion)
    expected_agreement = (np.sum(confusion, axis=1) * np.sum(confusion, axis=0)) / np.sum(confusion)
    kappa = (observed_agreement - expected_agreement) / (np.sum(confusion) - expected_agreement)
    return kappa


# 在 eval_semantic_segmentation 函数中添加这些指标的计算
def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1))

    f1 = calc_semantic_segmentation_f1(confusion)
    precision = calc_semantic_segmentation_precision(confusion)
    recall = calc_semantic_segmentation_recall(confusion)
    kappa = calc_semantic_segmentation_kappa(confusion)

    return {
        'iou': iou,
        'miou': np.nanmean(iou),
        'OA': pixel_accuracy,
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': np.nanmean(class_accuracy),
        'f1': np.nanmean(f1),
        'precision': np.nanmean(precision),
        'recall': np.nanmean(recall),
        'kappa': np.nanmean(kappa)
    }