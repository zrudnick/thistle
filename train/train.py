
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from train_base.model import *
from train.utils import *
from constants import *
import time


def initialize_model_and_optim(device, epochs, scheduler):
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L = 32
    N_GPUS = 2
    # W = np.asarray([11, 11, 11, 11])
    # AR = np.asarray([1, 1, 1, 1])
    W = np.asarray([11, 11, 11, 11, 11, 11, 11]) # formula: CL = 2 * np.sum(AR*(W-1))
    AR = np.asarray([1, 1, 1, 1, 1, 1, 4])
    BATCH_SIZE = 12*N_GPUS
    # if int(flanking_size) == 80:
    #     W = np.asarray([11, 11, 11, 11])
    #     AR = np.asarray([1, 1, 1, 1])
    #     BATCH_SIZE = 18*N_GPUS
    # elif int(flanking_size) == 400:
    #     W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
    #     AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    #     BATCH_SIZE = 18*N_GPUS
    # elif int(flanking_size) == 2000:
    #     W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
    #                     21, 21, 21, 21])
    #     AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
    #                     10, 10, 10, 10])
    #     BATCH_SIZE = 12*N_GPUS
    # elif int(flanking_size) == 10000:
    #     W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
    #                     21, 21, 21, 21, 41, 41, 41, 41])
    #     AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
    #                     10, 10, 10, 10, 25, 25, 25, 25])
    #     BATCH_SIZE = 6*N_GPUS
    CL = 2 * np.sum(AR*(W-1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    model = thistle(L, W, AR).to(device)
    print(model, file=sys.stderr)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)
    if scheduler == "MultiStepLR":
        scheduler_obj = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs-5, epochs-4, epochs-3, epochs-2, epochs-1], gamma=0.5)
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler_obj, params


def train(args):
    print("Running thistle with 'train' mode")
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim(device, args.epochs, args.scheduler)
    params["RANDOM_SEED"] = args.random_seed
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, test_h5f, 
                train_idxs, val_idxs, test_idxs, model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    test_h5f.close()