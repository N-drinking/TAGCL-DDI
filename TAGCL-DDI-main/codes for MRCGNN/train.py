import csv
import time
import torch
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score, f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve,auc
import random
import copy
from torch_geometric.utils import to_dense_adj
import numpy as np
torch.autograd.set_detect_anomaly(True)
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

def test(model, view_ncla, loader, data_o, args, printfou):
    if args.cuda:
        model.to('cuda')
        view_ncla.to('cuda')
        data_o.to('cuda')

    model.eval()
    y_pred = []
    y_label = []

    all_inp = []
    with torch.no_grad():
        for i, (inp) in enumerate(loader):
            heads, data_s, attention = view_ncla(args, data_o, True)
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()



            output, log = model(data_o, data_s, inp)
            log = torch.squeeze(log)


            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + log.flatten().tolist()
            all_inp.extend(inp)

    y_pred_train1 = []
    y_label_train = np.array(y_label)
    y_pred_train = np.array(y_pred).reshape((-1, 65))       # (np.max(y_pred_train,axis=1)).argsort()

    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    y_label_train1 = np.zeros((y_label_train.shape[0], 65))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    auc_hong=0
    aupr_hong=0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):

        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:

            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)

    return acc, f1_score1, recall1, precision1
