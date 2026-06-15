import torch


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def f1_score(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (2 * precision * recall / (precision + recall)).item()


def precision(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    return (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0


def recall(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    return (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0


def spearman_corr(preds, labels):
    preds = preds.float()
    labels = labels.float()
    n = preds.numel()
    preds_rank = preds.argsort().argsort().float()
    labels_rank = labels.argsort().argsort().float()
    cov = ((preds_rank - preds_rank.mean()) * (labels_rank - labels_rank.mean())).sum()
    std_p = ((preds_rank - preds_rank.mean()) ** 2).sum().sqrt()
    std_l = ((labels_rank - labels_rank.mean()) ** 2).sum().sqrt()
    if std_p * std_l == 0:
        return 0.0
    return (cov / (std_p * std_l)).item()


def matthews_corr(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    num = (tp * tn - fp * fn)
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()
    return (num / den).item() if den > 0 else 0.0
