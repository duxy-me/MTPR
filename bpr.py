import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from util import Metric
from model import Model


class BPR(Model):
    def __init__(self, ds, args, logging):
        self.ds = ds
        self.args = args
        self.logging = logging

        self.P = torch.nn.Embedding(self.ds.usz, self.ds.dim)
        pv = np.random.randn(self.ds.usz, self.ds.dim) * 0.01
        self.P.weight.data.copy_(torch.tensor(pv))
        self.P.weight.requires_grad = True

        self.Q = torch.nn.Embedding(self.ds.isz, self.ds.dim)
        qv = np.random.randn(self.ds.isz, self.ds.dim) * 0.01
        qv[list(self.ds.cold_start), :] = 0
        self.Q.weight.data.copy_(torch.tensor(qv))
        self.Q.weight.requires_grad = True

    def predict(self, uid, iid):
        return torch.sum(self.P(uid) * self.Q(iid), dim=1)

    def bpr_loss(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        return torch.sum(F.softplus(-result))

    def regs(self, uid, iid, niid):
        lr1, wd1 = self.args.p_emb
        p = self.P(uid)
        q = self.Q(iid)
        qn = self.Q(niid)
        emb_regs = torch.sum(p*p) + torch.sum(q*q) + torch.sum(qn*qn)
        return wd1 * emb_regs

    def train(self):
        lr1, wd1 = self.args.p_emb
        optimizer = optim.Adagrad([self.P.weight, self.Q.weight], lr=lr1, weight_decay=0)

        epochs = 100
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample()
            while True:
                optimizer.zero_grad()
                s = next(generator)
                if s is None:
                    break
                uid, iid, niid = s[:, 0], s[:, 1], s[:, 2]

                loss = self.bpr_loss(uid, iid, niid) + self.regs(uid, iid, niid)

                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                self.logging.info(
                    ["Epoch %d:" % epoch, torch.norm(self.P.weight).item(), torch.norm(self.Q.weight).item()])
                self.val(), self.test(), self.test_warm(), self.test_cold()

        self.logging.info(["final:", torch.norm(self.P.weight).item(), torch.norm(self.Q.weight).item()])
        self.test(), self.test_warm(), self.test_cold()

    def save(self, filename):
        np.save(filename, [self.P.weight.data.numpy(),
                           self.Q.weight.data.numpy()])