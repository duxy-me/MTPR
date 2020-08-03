
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score, \
    recall_score, precision_score, average_precision_score


class Metric:
    def get_annos(gt, preds):
        p_num = np.sum(gt > 0, axis=1, keepdims=True).flatten()
        #print(p_num)
        pos = np.argsort(-preds)[range(len(p_num)), p_num]
        #print(pos)
        ref_score = preds[range(len(pos)), pos]
        #print(preds.T, ref_score)
        annos = (preds.T - ref_score).T > 0
        return annos

    def ndcg(gt, preds):
        print('.ndcg')
        K = [5, 10, 20, 50, 100, 150, 200]
        return [ndcg_score(gt, preds, k=k) for k in K]

    def auc(gt, preds):
        print('.auc')
        return roc_auc_score(gt, preds, average='samples')


if __name__ == '__main__':
    gt = np.array([[1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]])
    a = np.array([[0, 0, 1, 3, 2, 0, 1], [0, 4, 3, 1, 2, 0, 1]])
    Metric.recall(gt, a), Metric.precision(gt, a), Metric.ap(gt, a)