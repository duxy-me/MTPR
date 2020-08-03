# from prepare import *

#VBPR

import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from util import Metric
from model import Model

class VBPR(Model):
    def __init__(self, ds, args, logging):
        self.ds = ds
        self.args = args
        self.logging = logging
        p_weight, q_weight = np.load('weights/%s_bpr_best.npy' % ds.name, allow_pickle=True)

        self.P = torch.nn.Embedding(self.ds.usz, self.ds.dim)
        self.P.weight.data.copy_(torch.tensor(p_weight))
        self.P.weight.requires_grad = True

        self.P2 = torch.nn.Embedding(self.ds.usz, self.ds.dim)
        self.P2.weight.data.normal_(0,0.01)
        self.P2.weight.requires_grad = True

        self.Q = torch.nn.Embedding(self.ds.isz, self.ds.dim)
        self.Q.weight.data.copy_(torch.tensor(q_weight))
        self.Q.weight.requires_grad = True

        self.W = torch.randn(self.ds.fsz,self.ds.dim, dtype=torch.float32) * 0.01
        self.W.requires_grad = True

        self.logging.info (['P.weight.dtype', self.P.weight.dtype])


    def predict(self, uid, iid):
        p1 = torch.sum(self.P(uid) * self.Q(iid), dim=1)
        p2 = torch.sum(self.P2(uid) * torch.mm(self.ds.newf[iid],self.W), dim = 1)
        return p1+p2

    def bpr_loss(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss


    def regs(self, uid, iid, niid):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx

        p1 = self.P(uid)
        p2 = self.P2(uid)
        q = self.Q(iid)
        qn = self.Q(niid)
        w = self.W
        emb_regs = torch.sum(p1*p1) + torch.sum(p2*p2) + torch.sum(q*q) + torch.sum(qn*qn)
        ctx_regs = torch.sum(w*w)

        return wd1 * emb_regs + wd2 * ctx_regs

    def train(self):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        optimizer = optim.Adagrad([self.P.weight,self.P2.weight, self.Q.weight], lr=lr1, weight_decay=0)
        optimizer2 = optim.Adam([self.W], lr=lr2, weight_decay=0)
        epochs = 40
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample()
            while True:
                optimizer.zero_grad()
                optimizer2.zero_grad()
                s = next(generator)
                if s is None:
                    break
                uid, iid, niid = s[:,0], s[:,1], s[:,2]

                loss = self.bpr_loss(uid, iid, niid) + self.regs(uid, iid, niid)

                loss.backward()

                optimizer.step()
                optimizer2.step()
            if epoch>0 and epoch % 20 == 0:
                self.logging.info(["Epoch %d:" % epoch,
                              torch.norm(self.P.weight),
                              torch.norm(self.P2.weight),
                              torch.norm(self.Q.weight),
                              torch.norm(self.W)])
                self.val(), self.test(), self.test_warm(), self.test_cold()
        self.logging.info(["final:",
                      torch.norm(self.P.weight),
                      torch.norm(self.P2.weight),
                      torch.norm(self.Q.weight),
                      torch.norm(self.W)])
        self.test(), self.test_warm(), self.test_cold()

    def save(self, filename):
        np.save(filename, [self.P.weight.data.numpy(),
                           self.P2.weight.data.numpy(),
                           self.Q.weight.data.numpy(),
                           self.W.data.numpy()])
        self.logging.info('weights are saved to ' + filename)
