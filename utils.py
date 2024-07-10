import os
import torch
import argparse

import models
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name'  , help='experiement name', type=str  , default='EXPFL')
    parser.add_argument('--model'     , help='model'           , type=str  , default='CNN')
    parser.add_argument('--dataset'   , help='dataset'         , type=str  , default='MNIST')
    parser.add_argument('--optimizer' , help='optimizer'       , type=str  , default='SGD')
    parser.add_argument('--lr'        , help='learning rate'   , type=float, default=1e-3)
    parser.add_argument('--decay'     , help='weight decay'    , type=float, default=1e-4)
    parser.add_argument('--batch_size', help='batch size'      , type=int  , default=64)
    parser.add_argument('--seed'      , help='random seed'     , type=int  , default=0)
    args = parser.parse_args()
    return args
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_model(args):
    return getattr(models, args.model)(n_class=cfg.N_CLASS[args.dataset])  

def build_criterion(args):
    return getattr(torch.nn, args.loss)()

def build_optimizer(model, args):
    return getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.decay)

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
