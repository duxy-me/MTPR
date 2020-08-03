from collections import OrderedDict
import numpy as np
from util import Metric
import torch
class Model:

    def compute_results(self, u, test_samples):
        rs = []
        for i in test_samples.T:
            rs.append(self.predict(u, torch.LongTensor(i)).detach().numpy())
        results = np.vstack(rs).T
        if np.isnan(results).any():
            raise Exception('nan')
        return results

    def compute_scores(self, gt, preds):
        ret = {
            'ndcg': Metric.ndcg(gt, preds),
            'auc': Metric.auc(gt, preds)
        }
        return ret

    def __logscore(self, scores):
        metrics = list(scores.keys())
        metrics.sort()
        self.logging.info(' '.join(['%s: %s' % (m,str(scores[m])) for m in metrics]))

    def test(self):
        u = torch.LongTensor(range(self.ds.usz))
        results = self.compute_results(u, self.ds.test_samples)
        scores = self.compute_scores(self.ds.test_gt, results)
        self.logging.info('----- test -----')
        self.__logscore(scores)

    def val(self):
        u = torch.LongTensor(range(self.ds.usz))
        results = self.compute_results(u, self.ds.val_samples)
        scores = self.compute_scores(self.ds.val_gt, results)
        self.logging.info('----- val -----')
        self.__logscore(scores)

    def test_warm(self):
        u = self.ds.test_warm_u
        results = self.compute_results(u, self.ds.test_warm_samples)
        scores = self.compute_scores(self.ds.test_warm_gt, results)
        self.logging.info('----- test_warm -----')
        self.__logscore(scores)

    def test_cold(self):
        u = self.ds.test_cold_u
        results = self.compute_results(u, self.ds.test_cold_samples)
        scores = self.compute_scores(self.ds.test_cold_gt, results)
        self.logging.info('----- test_cold -----')
        self.__logscore(scores)

    def train(self):
        raise Exception('no implementation')

    def regs(self):
        raise Exception('no implementation')

    def predict(self):
        raise Exception('no implementation')

    def save(self):
        raise Exception('no implementation')