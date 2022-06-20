import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def get_acc(predict, target):
    predict = predict.detach().cpu().squeeze().numpy()
    target = target.detach().cpu().squeeze().numpy()
    acc = np.sum(predict == target) / len(predict)
    return acc


def get_metric(predict_score, target_label):
    # predict_score : numpy array, shape k
    # target_label : numpy array, shape k
    auroc = roc_auc_score(target_label, predict_score)
    aupr_in = average_precision_score(target_label, predict_score)
    aupr_out = average_precision_score(1 - target_label, -predict_score)
    fpr, tpr, thresh = roc_curve(target_label, predict_score)
    index95 = int(np.argmin(np.abs(np.array(tpr) - 0.95)))
    fpr95, thresh95 = fpr[index95], thresh[index95]
    return auroc, aupr_in, aupr_out, fpr95, thresh95


if __name__ == '__main__':
    k = 100
    x = np.random.uniform(0, 1, size=k)
    y = np.random.uniform(0, 1, size=k)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    print(get_metric(x, y))