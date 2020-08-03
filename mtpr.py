# from prepare import *

import torch.optim as optim
import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from util import Metric
from model import Model


class MTPR(Model):
    def __init__(self, ds, args, logging):
        self.ds = ds
        self.args = args
        self.logging = logging

        p1_weight = np.random.randn(self.ds.usz, self.ds.dim) * 0.01
        q_weight = np.random.randn(self.ds.usz, self.ds.dim) * 0.01
        self.logging.info('p/q weights are randomly initialized ')

        # uncomment to load the pretrained bpr parameters
        # p1_weight, q_weight = np.load('weights/%s_bpr_best.npy' % ds.name, allow_pickle=True)
        # self.logging.info('loaded p/q weights from weights/%s_bpr_best.npy')

        p2_weight = np.random.randn(self.ds.usz, self.ds.dim) * 0.01
        p_weight = np.concatenate([p1_weight,p2_weight], axis=1)

        self.P = torch.nn.Embedding(self.ds.usz, self.ds.dim * 2)
        self.P.weight.data.copy_(torch.tensor(p_weight))
        self.P.weight.requires_grad = True

        self.Q = torch.nn.Embedding(self.ds.isz, self.ds.dim)
        self.Q.weight.data.copy_(torch.tensor(q_weight))
        self.Q.weight.requires_grad = True

        self.W = torch.randn(self.ds.fsz,self.ds.dim, dtype=torch.float32) * 0.01
        self.W.requires_grad = True
        self.weu = torch.randn(self.ds.dim * 2,self.ds.dim, dtype=torch.float32) * 0.01
        self.weu.requires_grad = True
        self.wei = torch.randn(self.ds.dim * 2,self.ds.dim, dtype=torch.float32) * 0.01
        self.wei.requires_grad = True

    def fimg(self, iid):    # normal representation
        return torch.cat((self.Q(iid), torch.mm(self.ds.newf[iid],self.W)), dim = 1)

    def zimg(self, iid):    # conterfactual representation
        fzero = torch.zeros_like(self.Q(iid))
        return torch.cat((fzero, torch.mm(self.ds.newf[iid],self.W)), dim = 1)

    def trf(self, emb, theta):
        return torch.mm(emb, theta)

    def predict(self, uid, iid):
        return torch.sum(self.trf(self.P(uid), self.weu) * self.trf(self.fimg(iid), self.wei), dim=1)

    def predict_z(self, uid, iid):
        return torch.sum(self.trf(self.P(uid), self.weu) * self.trf(self.zimg(iid), self.wei), dim=1)

    def bpr_loss_i(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss
   
    def bpr_loss_f(self, uid, iid, niid):
        pred_p = self.predict_z(uid, iid)
        pred_n = self.predict_z(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_if(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict_z(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss


    def bpr_loss_fi(self, uid, iid, niid):
        pred_p = self.predict_z(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    # multi-task learning
    def test_loss(self, uid, iid, niid):
        aloss = 0
        aloss += self.bpr_loss_i(uid, iid, niid) + self.bpr_loss_f(uid,iid,niid)   # two crucial task
        aloss += self.bpr_loss_if(uid, iid, niid) + self.bpr_loss_fi(uid,iid,niid) # two constraints
        return aloss

    def regs(self, uid, iid, niid):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        lr3, wd3 = self.args.p_proj

        p = self.P(uid)
        q = self.Q(iid)
        qn = self.Q(niid)
        w = self.W
        weu = self.weu
        wei = self.wei
        emb_regs = torch.sum(p*p) + torch.sum(q*q) + torch.sum(qn*qn)
        ctx_regs = torch.sum(w*w) + torch.sum(weu*weu)
        proj_regs = torch.sum(wei*wei)

        return wd1 * emb_regs + wd2 * ctx_regs + wd3 * proj_regs

    def train(self):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        lr3, wd3 = self.args.p_proj
        optimizer = optim.Adagrad([self.P.weight,self.Q.weight], lr=lr1, weight_decay=0)
        optimizer2 = optim.Adam([self.W, self.weu], lr=lr2, weight_decay=0)
        optimizer3 = optim.Adam([self.wei], lr=lr3, weight_decay=0)
        epochs = 100
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample()
            while True:
                s = next(generator)
                if s is None:
                    break
                uid, iid, niid = s[:,0], s[:,1], s[:,2]

                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                aloss = self.test_loss(uid,iid,niid) + self.regs(uid, iid, niid)
                aloss.backward()

                optimizer.step()
                optimizer2.step()
                optimizer3.step()

            if epoch > 0 and epoch % 10 == 0:
                self.logging.info(["Epoch %d:" % epoch,
                      torch.norm(self.P.weight).item(),
                      torch.norm(self.Q.weight).item(),
                      torch.norm(self.W).item(),
                      torch.norm(self.weu).item(),
                      torch.norm(self.wei).item()])
                self.val(), self.test(), self.test_warm(), self.test_cold()

        self.logging.info(["final",
                       torch.norm(self.P.weight).item(),
                       torch.norm(self.Q.weight).item(),
                       torch.norm(self.W).item(),
                       torch.norm(self.weu).item(),
                       torch.norm(self.wei).item()])
        self.val(), self.test(), self.test_warm(), self.test_cold()

    def save(self, filename):
        np.save(filename, [self.P.weight.data.numpy(),
                           self.Q.weight.data.numpy(),
                           self.W.data.numpy(),
                           self.weu.data.numpy(),
                           self.wei.data.numpy()])
        self.logging.info('weights are saved to ' + filename)