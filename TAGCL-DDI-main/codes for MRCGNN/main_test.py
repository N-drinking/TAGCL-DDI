import parser
import torch
import numpy as np
from train import test
from parms_setting import settings
from data_preprocess import load_data
from instantiation import Create_model
import os
import random
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
set_random_seed(1, deterministic=False)

# parameters setting
args = settings()
args.cuda = not args.no_cuda and torch.cuda.is_available()


data_o, train_loader, val_loader, test_loader = load_data(args)


# train and test model
model, view_ncla, optimizer, scheduler = Create_model(args, data_o)
zhongzi = args.zhongzi
model_max = torch.load('data/'+str(zhongzi)+'/model.pth')

# test(model_max, view_ncla, view_gen1, view_gen2, optimizer, scheduler, data_o, train_loader, val_loader, test_loader, args)
acc_test, f1_test, recall_test,precision_test= test(model_max, view_ncla, test_loader, data_o, args, 1)
print( 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test),'recall_test: {:.4f}'.format(recall_test))
