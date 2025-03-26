import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from constants import *
import time
from train_base.model import *
from train.utils import *

def initialize_model_and_optim(device, flanking_size, pretrained_model, random_seed):
    L = 32
    N_GPUS = 2
    # W = np.asarray([11, 11, 11, 11])
    # AR = np.asarray([1, 1, 1, 1])
    W = np.asarray([11, 11, 11, 11, 11, 11, 11]) # formula: CL = 2 * np.sum(AR*(W-1))
    AR = np.asarray([1, 1, 1, 1, 1, 1, 4])
    BATCH_SIZE = 12*N_GPUS  
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    
    # Initialize the model
    model = thistle(L, W, AR).to(device)

    # Print the shapes of the parameters in the initialized model
    print("\nInitialized model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    print(model, file=sys.stderr)    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS, 'RANDOM_SEED': random_seed}
    return model, optimizer, scheduler, params


def test(args):
    print("Running thistle with 'test' mode")
    device = setup_environment(args)
    log_output_test_base = initialize_test_paths(args)
    test_h5f = load_test_datasets(args)    
    test_idxs = generate_test_indices(args.random_seed, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model, args.random_seed)
    test_metric_files = create_metric_files(log_output_test_base)
    test_model(model, optimizer, test_h5f, test_idxs, args, device, params, test_metric_files)
    test_h5f.close()