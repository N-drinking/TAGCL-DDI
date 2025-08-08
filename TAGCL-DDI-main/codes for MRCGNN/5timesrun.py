# import pandas as pd
# import numpy as py
# import random
import os
import torch
import random

import copy


import numpy as np
def set_random_seed(seed=1, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for k in range(5):
    os.system('python main_test.py --out_file test.txt --zhongzi '+str(k))


# loss_ncla_range = [0.4, 0.5, 0.6, 0.7,1]  # 学习率的范围，可以根据需要调整
#
# for loss_ncla in loss_ncla_range:
#      os.system('python main.py --out_file test.txt --loss_ncla ' + str(loss_ncla) + ' --zhongzi '+str(0))
# loss_autoGCL_range = [0.0005, 0.005, 0.05, 0.5,1]  # 学习率的范围，可以根据需要调整
#
# for loss_autoGCL in loss_autoGCL_range:
#      os.system('python main.py --out_file test.txt --loss_autoGCL ' + str(loss_autoGCL) + ' --zhongzi '+str(0))




