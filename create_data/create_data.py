import h5py
import numpy as np
from tqdm import tqdm
import time
import argparse         
import os
import random
from constants import *
from create_data.utils import ceil_div, replace_non_acgt_to_n, create_datapoints
import create_data.utils as utils

CHUNK_SIZE = 100 # size of chunks to process data in

def write_h5_file(output_dir, data_type, data):
    """
    Write the data to an h5 file.
    """
    if output_dir[-1] == "/":
        h5fname = output_dir + f'datafile_{data_type}.h5'
    else: h5fname = output_dir + f'/datafile_{data_type}.h5'
    h5f = h5py.File(h5fname, 'w')
    dt = h5py.string_dtype(encoding='utf-8')
    
    #dataset_names = ['NAME', 'CHROM', 'SEQ', 'LABEL']
    dataset_names = ['SEQ', 'LABEL']
    for i, name in enumerate(dataset_names):
        h5f.create_dataset(name, data=np.asarray(data[i], dtype=dt), dtype=dt)    
    h5f.close()

def get_sequences_and_labels(data, labels, output_dir, train_or_test):
    """
    Copy my data file into pre-h5ad format
    """
    #fw_stats = open(f"{output_dir}{train_or_test}_stats.txt", "w")
    
    NAME = []      # Gene Name
    CHROM = []     # Chromosome
    SEQ = []       # Nucleotide sequence
    LABEL = []     # Label for each nucleotide in the sequence
    #GENE_COUNTER = 0

    with open(data, "r") as X, open(labels, "r") as y:
        y_lines = y.readlines()

        for (i, x_line) in enumerate(X):
            SEQ.append(x_line[:-1])         # remove \n
            LABEL.append(y_lines[i][0])

        #fw_stats.write(f"{gene.seqid}\t{gene.start}\t{gene.end}\t{gene.id}\t{1}\t{gene.strand}\n")
        #utils.check_and_count_motifs(gene_seq, labels, donor_motif_counts, acceptor_motif_counts)

    # Split 80:20 into train and test data
    i = len(SEQ)//5 * 4
    train_set = [SEQ[:i], LABEL[:i]]
    test_set = [SEQ[i:], LABEL[i:]]

    combined_train = list(zip(train_set[0], train_set[1]))
    random.shuffle(combined_train)
    unique_train = list(set(combined_train))
    SEQ, LABEL = zip(*unique_train)
    train_set = [SEQ, LABEL]

    combined_test = list(zip(test_set[0], test_set[1]))
    random.shuffle(combined_test)
    unique_test = list(set(combined_test))
    SEQ, LABEL = zip(*unique_test)
    test_set = [SEQ, LABEL]

    
    if train_or_test == 'train-test':
        print("Length of train data: ", len(train_set[0]))
        print("Length of test data: ", len(test_set[0]))
        return train_set, test_set
        
    else: # test
        print("Length of test data: ", len(test_set[0]))
        return test_set


def create_datafile(args):
    print("--- Step 1: Creating datafile.h5 ... ---")
    start_time = time.process_time()

    # Collect sequences and labels for testing and/or training groups
    if args.chr_split == 'test':
        print("Creating test datafile...")
        test_data = get_sequences_and_labels(args.data, args.labels, args.output_dir, 'test')
        write_h5_file(args.output_dir, "test", test_data)

    elif args.chr_split == 'train-test':

        print("Creating train and test datafiles...")
        train_data, test_data = get_sequences_and_labels(args.data, args.labels, args.output_dir, 'train-test')

        #print("Creating test datafile...")
        #test_data = get_sequences_and_labels(args.data, args.labels, args.output_dir, 'test')
        
        # Write the data to h5 files
        write_h5_file(args.output_dir, "train", train_data)
        write_h5_file(args.output_dir, "test", test_data)   


    print("--- %s seconds ---" % (time.process_time() - start_time))


def create_dataset(args):
    print("--- Step 2: Creating dataset.h5 ... ---")
    start_time = time.process_time()
    
    dataset_ls = [] 
    if args.chr_split == 'test':
        dataset_ls.append('test')

    elif args.chr_split == 'train-test':
        dataset_ls.append('test')
        dataset_ls.append('train')

    for dataset_type in dataset_ls:
        input_file = f"{args.output_dir}/datafile_{dataset_type}.h5"
        output_file = f"{args.output_dir}/dataset_{dataset_type}.h5"

        print(f"\tReading {input_file} ... ")
        with h5py.File(input_file, 'r') as h5f:
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]

        print(f"\tWriting {output_file} ... ")
        with h5py.File(output_file, 'w') as h5f2:
            seq_num = len(SEQ)
            # create dataset
            num_chunks = ceil_div(seq_num, CHUNK_SIZE)
            for i in tqdm(range(num_chunks), desc='Processing chunks...'):
                # each dataset has CHUNK_SIZE genes
                if i == num_chunks - 1: # if last chunk, process remainder or full chunk size if no remainder
                    NEW_CHUNK_SIZE = seq_num % CHUNK_SIZE or CHUNK_SIZE 
                else:
                    NEW_CHUNK_SIZE = CHUNK_SIZE
                X_batch, Y_batch = [], [[] for _ in range(1)]
                for j in range(NEW_CHUNK_SIZE):
                    idx = i*CHUNK_SIZE + j
                    seq_decode = SEQ[idx].decode('ascii')
                    
                    label_decode = LABEL[idx].decode('ascii')
                    fixed_seq = replace_non_acgt_to_n(seq_decode)
                    X, Y = create_datapoints(fixed_seq, label_decode)
                    X_batch.extend(X)
                    Y_batch[0].extend(Y[0])
                # Convert batches to arrays and save as HDF5
                X_batch = np.asarray(X_batch).astype('int8')
                Y_batch[0] = np.asarray(Y_batch[0]).astype('int8')

                h5f2.create_dataset('X' + str(i), data=X_batch)
                h5f2.create_dataset('Y' + str(i), data=Y_batch)
    print("--- %s seconds ---" % (time.process_time() - start_time))