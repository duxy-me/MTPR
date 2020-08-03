import argparse
import sys
import logging
from amazon.load_data import dataset as ds_amazon

from bpr import BPR
from vbpr import VBPR
from cbpr import CBPR
from mtpr import MTPR
from amr import AMR



def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for models')
    parser.add_argument('--dataset', nargs='?', default='amazon',
                        help='one of amazon')
    parser.add_argument('--model', nargs='?', default='mtpr',
                        help='one of bpr, vbpr, cbpr, amr, mtpr')
    parser.add_argument('--p_emb', nargs='?', default='[0.01, 0]',
                        help='lr and reg for id embeddings')
    parser.add_argument('--p_ctx', nargs='?', default='[0.01, 0.01]',
                        help='lr and reg for context features')
    parser.add_argument('--p_proj', nargs='?', default='[0.01, 0.01]',
                        help='lr and reg for wei only')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='epsilong for noises')
    parser.add_argument('--lmd', type=float, default=1,
                        help='balance the adv')
    parser.add_argument('--tolog', type=int, default=1,
                        help='0: output to stdout, 1: output to logfile')
    parser.add_argument('--bsz', type=int, default=512,
                        help='batch size')
    parser.add_argument('--ssz', type=int, default=1000,
                        help='size of test samples, including positive and negative samples')
    return parser.parse_args()

args = parse_args()

args.p_emb = eval(args.p_emb)
args.p_ctx = eval(args.p_ctx)
args.p_proj = eval(args.p_proj)

if args.tolog == 0:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
else:
    logfilename = 'logs/%s_%s_%s_%s_%s.log' % (args.dataset, args.model, str(args.p_emb), str(args.p_ctx), str(args.p_proj))
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
    logging.info('log info to ' + logfilename)

logging.info(args)
if args.dataset == 'amazon':
    ds = ds_amazon(logging, args)
else:
    raise Exception('no dataset' + args.dataset)

if args.model == 'bpr':
    model = BPR(ds,args, logging)
elif args.model == 'cbpr':
    model = CBPR(ds,args, logging)
elif args.model == 'vbpr':
    model = VBPR(ds,args, logging)
elif args.model == 'amr':
    model = AMR(ds,args, logging)
elif args.model == 'mtpr':
    model = MTPR(ds,args, logging)
else:
    raise Exception('unknown model type', args.model)

model.train()

weight_filename = 'weights/%s_%s_%s_%s_%s.npy' % (args.dataset, args.model, str(args.p_emb), str(args.p_ctx), str(args.p_proj))
model.save(weight_filename)