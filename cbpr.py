# from prepare import *

#CBPR

import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from util import Metric
from model import Model

class CBPR(Model):
    def __init__(self, ds, args, logging):
        self.ds = ds
        self.args = args
        self.logging = logging
        self.P2 = torch.nn.Embedding(self.ds.usz, self.ds.dim)
        self.P2.weight.data.normal_(0,0.01)

        self.W = torch.randn(self.ds.fsz,self.ds.dim, dtype=torch.float32) * 0.01

        self.P2.weight.requires_grad = True
        self.W.requires_grad = True


    def predict(self, uid, iid):
        p2 = torch.sum(self.P2(uid) * torch.mm(self.ds.newf[iid],self.W), dim = 1)
        return p2

    def bpr_loss(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss


    def train(self):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        optimizer = optim.Adagrad([self.P2.weight], lr=lr1, weight_decay=wd1)
        optimizer2 = optim.Adam([self.W], lr=lr2, weight_decay=wd2)
        epochs = 100
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample()
            while True:
                optimizer.zero_grad()
                optimizer2.zero_grad()
                s = next(generator)
                if s is None:
                    break
                uid, iid, niid = s[:,0], s[:,1], s[:,2]

                loss = self.bpr_loss(uid,iid,niid)
                loss.backward()

                optimizer.step()
                optimizer2.step()
            if epoch % 5 == 0:
                self.logging.info(["Epoch %d:" % epoch,
                              torch.norm(self.P2.weight),
                              torch.norm(self.W)])
                self.val(), self.test(), self.test_warm(), self.test_cold()

        self.logging.info(["final:",
                      torch.norm(self.P2.weight),
                      torch.norm(self.W)])
        self.test(), self.test_warm(), self.test_cold()

    def save(self, filename):
        np.save(filename, [self.P2.weight.data.numpy(),
                           self.W.data.numpy()])
        self.logging.info('weights are saved to ' + filename)
