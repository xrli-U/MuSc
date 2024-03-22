import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def compute_metrics(gt_sp=None, pr_sp=None, gt_px=None, pr_px=None):
    # classification
    if gt_sp is None or pr_sp is None or gt_sp.sum() == 0 or gt_sp.sum() == gt_sp.shape[0]:
        auroc_sp, f1_sp, ap_sp = 0, 0, 0
    else:
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

    # segmentation
    if gt_px is None or pr_px is None or gt_px.sum() == 0:
        auroc_px, f1_px, ap_px, aupro = 0, 0, 0, 0
    else:
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        aupro = cal_pro_score(gt_px.squeeze(), pr_px.squeeze())

    image_metric = [auroc_sp, f1_sp, ap_sp]
    pixel_metric = [auroc_px, f1_px, ap_px, aupro]

    return image_metric, pixel_metric
    