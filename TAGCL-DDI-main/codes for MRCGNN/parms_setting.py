import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import torch
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
def settings():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')
    # '''
    parser.add_argument('--out_file', required=True,default='result.txt',
                        help='Path to data result file. e.g., result.txt')
    # '''
    # '''
    parser.add_argument('--lr', type=float, default=15e-4,
                        help='Initial learning rate. Default is 5e-4.')
    # '''
#
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')

    parser.add_argument('--weight_decay', default=1e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')
#
    parser.add_argument('--batch', type=int, default=4096,
                        help='Batch size. Default is 256.')
#
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train. Default is 30.')

    parser.add_argument('--network_ratio', type=float, default=1,
                        help='Remain links in network. Default is 1')

    parser.add_argument('--loss_mlp', type=float, default=1,
                        help='Ratio of task1. Default is 1')
###
    parser.add_argument('--loss_ncla', type=float, default=0.5,
                        help='Ratio of task2. Default is 0.1')
##
    parser.add_argument('--loss_autoGCL', type=float, default=0.5,
                        help='Ratio of task3. Default is 0.1')
#
    # GCN parameters#
    parser.add_argument('--dimensions', type=int, default=128,
                        help='dimensions of feature. Default is 128.')

    parser.add_argument('--hidden1', default=64,
                        help='Number of hidden units for encoding layer 1 for CSGNN. Default is 64.')
#
    parser.add_argument('--hidden2', default=32,
                        help='Number of hidden units for encoding layer 2 for CSGNN. Default is 32.')

    parser.add_argument('--decoder1', default=512,
                        help='Number of hidden units for decoding layer 1 for CSGNN. Default is 512.')
    parser.add_argument('--zhongzi', default=0,
                        help='Number of zhongzi.')

    # add
    ''''''
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--out-dim", type=int, default=2,
                        help="number of hidden units")
    parser.add_argument("--num-head-Auto", type=int, default=1,
                        help="number of Auto head")
    parser.add_argument("--num-head-NCLA", type=int, default=4,
                        help="number of NCLA head")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.5,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")

    parser.add_argument("--epsilon", type=int, default=1e-7,
                        help="limit feature floor value.")
    parser.add_argument("--tau", type=float, default=1,
                        help="temperature-scales")
    ''''''

#
    args = parser.parse_args()

    return args
