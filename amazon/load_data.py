import numpy as np
import torch
from sklearn.decomposition import PCA


class dataset:
    def __init__(self, logging,args):
        self.logging = logging
        np.random.seed(1122)  # to ensure the code generate the same test sets.
        self.name = 'amazon'
        self.data = np.load('amazon/men.npy', allow_pickle=True).item()

        self.usz = len(self.data['train'])
        self.isz = len(self.data['feat'])
        self.dim = 32
        self.bsz = args.bsz
        self.ssz = args.ssz

        # training set
        d_train = self.data['train']
        self.train_list = []
        self.warm_start = set([])
        for user, items in enumerate(d_train):
            for item in items:
                self.train_list.append((user, item))
                self.warm_start.add(item)

        self.train = [1 for i in range(self.usz)]
        for user in range(self.usz):
            self.train[user] = set(d_train[user])

        self.logging.info('train list is ready')

        # val set
        self.posset = set(self.train_list)
        self.logging.info(['posset sized: ', len(self.posset)])
        self.cold_start = set([])
        d_val = self.data['val']
        val_list = [[] for i in range(self.usz)]
        self.val_gt = np.zeros((self.usz, self.ssz))
        for user, items in enumerate(d_val):
            val_list[user].extend(items)
            sits = set(items)
            psz = len(items)
            self.cold_start.update(items)
            self.val_gt[user, :psz] = 1
            for i in range(self.ssz - psz):
                ele = items[-1]
                while ele in sits or (user, ele) in self.posset:
                    ele = np.random.randint(self.isz)
                val_list[user].append(ele)
                sits.add(ele)
        self.val_samples = np.array(val_list)

        self.logging.info('val list is ready')
        # test set
        d_test = self.data['test']
        test_list = [[] for i in range(self.usz)]
        self.test_gt = np.zeros((self.usz, self.ssz))
        for user, items in enumerate(d_test):
            test_list[user].extend(items)
            sits = set(items)
            psz = len(items)
            self.cold_start.update(items)
            self.test_gt[user, :psz] = 1
            for i in range(self.ssz - psz):
                ele = items[-1]
                while ele in sits or (user, ele) in self.posset:
                    ele = np.random.randint(self.isz)
                test_list[user].append(ele)
                sits.add(ele)
        self.test_samples = np.array(test_list)
        self.logging.info('test list is ready')

        self.logging.info(['test items:', len(self.cold_start)])
        self.cold_start = self.cold_start - self.warm_start
        self.logging.info(['warm:', len(self.warm_start), 'cold:', len(self.cold_start)])

        # To ensure all the codes obtain the same test samples
        if np.sum(np.abs(self.test_samples[0,:9] - np.array([72987, 31858, 28654, 48602, 29071, 43490, 48855, 75002, 55736]))) != 0 or\
            np.sum(np.abs(self.test_samples[0,-10:] - np.array([39285, 34394, 86058, 29585, 32620, 74762, 14512, 23269, 51687, 15557]))) != 0:
            raise Exception('different test sets')

        self.train_list = np.array(self.train_list)
        self.sz = self.train_list.shape[0]

        # item attributes
        newf = self.data['feat']
        self.logging.info(newf.shape)
        self.newf = torch.Tensor(newf)
        self.fsz = 64
        self.logging.info('features have been preprocessed')

        # set the validation and test sets for warm and cold positive items respectively.
        self.val_warm_u, \
        self.val_cold_u, \
        self.val_warm_samples, \
        self.val_cold_samples, \
        self.val_warm_gt, \
        self.val_cold_gt = self.seperate(self.data['val'])
        self.logging.info('val warm/cold seperated')

        self.test_warm_u, \
        self.test_cold_u, \
        self.test_warm_samples, \
        self.test_cold_samples, \
        self.test_warm_gt, \
        self.test_cold_gt = self.seperate(self.data['test'])
        self.logging.info('test warm/cold seperated')

    def seperate(self, d_pos):
        warm_u = []
        cold_u = []
        warm_gt = []
        cold_gt = []
        warm_samples = []
        cold_samples = []

        for u in range(self.usz):
            for iid in d_pos[u]:
                cs = []
                ws = []
                if iid in self.cold_start:
                    cs.append(iid)
                else:
                    ws.append(iid)

            pcsz = len(cs)

            if pcsz > 0:
                cold_u.append(u)  # 1
                cold_gt.append(np.zeros(self.ssz))
                cold_gt[-1][:pcsz] = 1  # 2

                sits = set(d_pos[u])  # The set of selected items.
                for i in range(self.ssz - pcsz):
                    ele = np.random.randint(self.isz)
                    while ele in sits or (u, ele) in self.posset:
                        ele = np.random.randint(self.isz)
                    cs.append(ele)
                    sits.add(ele)
                cold_samples.append(cs)  # 3

            pwsz = len(ws)
            if pwsz > 0:
                warm_u.append(u)  # 1
                warm_gt.append(np.zeros(self.ssz))
                warm_gt[-1][:pwsz] = 1  # 2

                sits = set(d_pos[u])  # The set of selected items.
                for i in range(self.ssz - pwsz):
                    ele = np.random.randint(self.isz)
                    while ele in sits or (u, ele) in self.posset:
                        ele = np.random.randint(self.isz)
                    ws.append(ele)
                    sits.add(ele)
                warm_samples.append(ws)  # 3

        self.logging.info(list(map(lambda x: x.shape, [torch.tensor(warm_u), torch.tensor(cold_u),
                                                       torch.tensor(warm_samples), torch.tensor(cold_samples),
                                                       np.array(warm_gt), np.array(cold_gt)])))

        return torch.tensor(warm_u), torch.tensor(cold_u), \
               torch.tensor(warm_samples), torch.tensor(cold_samples), \
               np.array(warm_gt), np.array(cold_gt)

    def sample(self):
        # make pair
        np.random.shuffle(self.train_list)
        for i in range(self.sz // self.bsz):
            pairs = []
            sub_train_list = self.train_list[i * self.bsz:(i + 1) * self.bsz, :]
            for i, j in sub_train_list:
                i_neg = j
                while i_neg in self.train[i] or i_neg in self.cold_start:
                    i_neg = np.random.randint(self.isz)
                pairs.append((i, j, i_neg))
            yield torch.LongTensor(pairs)
        yield None

