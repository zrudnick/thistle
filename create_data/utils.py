###############################################################################
'''The file includes functions to process sequences to/from .h5 datasets.'''
###############################################################################

import os
import numpy as np
import torch
from math import ceil
import gffutils
import random
from sklearn.metrics import average_precision_score
from constants import *
#CL_max = 200
#SL = 3

def check_and_count_motifs(seq, labels, donor_motif_counts, acceptor_motif_counts):
    """
    Check sequences for donor and acceptor motifs and count their occurrences.

    Parameters:
    - seq: The DNA sequence (str).
    - labels: Array of labels indicating locations of interest in the sequence.
    """     
    for i, label in enumerate(labels):
        if label == 2:  # Donor site
            d_motif = str(seq[i+1:i+3])
            donor_motif_counts[d_motif] = donor_motif_counts.get(d_motif, 0) + 1
        elif label == 1:  # Acceptor site
            a_motif = str(seq[i-2:i])
            acceptor_motif_counts[a_motif] = acceptor_motif_counts.get(a_motif, 0) + 1


def print_motif_counts(donor_motif_counts, acceptor_motif_counts):
    """
    Print the counts of donor and acceptor motifs.
    """
    print("*******************")
    print("Splice site motif counts:")
    print("\tDonor motifs:")
    for motif, count in donor_motif_counts.items():
        print(f"\t{motif}: {count}")
    print("\n\tAcceptor motifs:")
    for motif, count in acceptor_motif_counts.items():
        print(f"\t{motif}: {count}")
    print("\nTotal donor motifs: ", sum(donor_motif_counts.values()))
    print("Total acceptor motifs: ", sum(acceptor_motif_counts.values()))
    print("*******************")
    
    
###################################################
# create_datafile.py functions
###################################################
def get_chromosome_lengths(db):
    """Extract all unique chromosomes and their lengths from the GFF database."""
    chromosomes = {}
    for feature in db.all_features():
        if feature.seqid not in chromosomes:
            chromosomes[feature.seqid] = feature.end
        else:
            chromosomes[feature.seqid] = max(chromosomes[feature.seqid], feature.end)
    return chromosomes


def split_chromosomes(db, method='random', split_ratio=0.8):
    chromosome_lengths = get_chromosome_lengths(db)
    """Split chromosomes into training and testing groups."""
    if method == 'random':
        total_length = sum(chromosome_lengths.values())
        target_train_length = total_length * split_ratio
        target_test_length = total_length * (1-split_ratio)
        chromosomes = list(chromosome_lengths.keys())
        random.shuffle(chromosomes)
        train_chroms = {}
        test_chroms = {}
        current_train_length = 0
        current_test_length = 0 
        
        for chrom in chromosomes:
            if current_test_length < target_test_length:
                test_chroms[chrom] = chromosome_lengths[chrom]
                current_test_length += chromosome_lengths[chrom]
            else:
                train_chroms[chrom] = chromosome_lengths[chrom]
                current_train_length += chromosome_lengths[chrom]
    elif method == 'human':
        # following SpliceAI default splitting for human chromosomes
        train_chroms = {
        'chr2': 0, 'chr4': 0, 'chr6': 0, 'chr8': 0, 
        'chr10': 0, 'chr11': 0, 'chr12': 0, 'chr13': 0,
        'chr14': 0, 'chr15': 0, 'chr16': 0, 'chr17': 0, 
        'chr18': 0, 'chr19': 0, 'chr20': 0, 'chr21': 0, 
        'chr22': 0, 'chrX': 0, 'chrY': 0
        }
        test_chroms = {
            'chr1': 0, 'chr3': 0, 'chr5': 0, 'chr7': 0, 'chr9': 0
        }
    else: 
        raise ValueError("Invalid chromosome split method. Use 'random' or 'human'.")
    return train_chroms, test_chroms


def create_or_load_db(gff_file, db_file='gff.db'):
    """
    Create a gffutils database from a GFF file, or load it if it already exists.

    Parameters:
    - gff_file: Path to GFF file
    - db_file: Path to save or load the database file (default: 'gff.db')

    Returns:
    - db: gffutils FeatureDB object
    """

    if not os.path.exists(db_file):
        print("Creating new database...")
        db = gffutils.create_db(gff_file, dbfn=db_file, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    else:
        print("Loading existing database...")
        db = gffutils.FeatureDB(db_file)
    return db

###################################################
# create_dataset.py functions
###################################################
# One-hot encoding of the inputs: 
# 1: A;  2: C;  3: G;  4: T;  0: padding
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
 #                    [0, 0, 0, 0, 1]])

# One-hot encoding of the outputs: 
# 0: no tis;  1: tis;
OUT_MAP = np.asarray([[0, 0],
                      [1, 0],
                      [0, 1]])


def ceil_div(x, y):
    """
    Calculate the ceiling of a division between two numbers.

    Parameters:
    - x (int): Numerator
    - y (int): Denominator

    Returns:
    - int: The ceiling of the division result.
    """
    return int(ceil(float(x)/y))


def one_hot_encode(Xd, Yd):
    """
    Perform one-hot encoding on both the input sequence data (Xd) and the output label data (Yd) using
    predefined mappings (IN_MAP for inputs and OUT_MAP for outputs).

    Parameters:
    - Xd (numpy.ndarray): An array of integers representing the input sequence data where each nucleotide
        is encoded as an integer (1 for 'A', 2 for 'C', 3 for 'G', 4 for 'T', and 0 for padding).
    - Yd (list of numpy.ndarray): A list containing a single array of integers representing the output label data,
        where each label is encoded as an integer (0 for 'no tis', 1 for 'tis', and -1 for padding).

    Returns:
    - numpy.ndarray: the one-hot encoded input sequence data.
    - numpy.ndarray: the one-hot encoded output label data.
    """
    
    # edit ATG to be special char
    # for i in range(100, 103):
    #     Xd[0][i] = 5.0
    X = IN_MAP[Xd.astype('int8')]
    Y = [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]

    return X, Y


def replace_non_acgt_to_n(input_string):
    """
    Use a generator expression to go through each character in the input string.
    If the character is in the set of allowed characters, keep it as is.
    Otherwise, replace it with 'N'.

    Parameters:
    - input_string (str): The nucleotide sequence.

    Returns:
    - str: The modified sequence with non-ACGT nucleotides replaced by 'N'.
    """
    # Define the set of allowed characters
    allowed_chars = {'A', 'C', 'G', 'T'}    
    return ''.join(char if char in allowed_chars else 'N' for char in input_string)


# so broken
def reformat_data(X0, Y0):
    """
    Reformat sequence and label data into fixed-size blocks for processing.
    This function converts X0, Y0 of the create_datapoints function into
    blocks such that the data is broken down into data points where the
    input is a sequence of length SL+CL_max corresponding to SL nucleotides
    of interest and CL_max context nucleotides, the output is a sequence of
    length SL corresponding to the splicing information of the nucleotides
    of interest. The CL_max context nucleotides are such that they are
    CL_max/2 on either side of the SL nucleotides of interest.

    Parameters:
    - X0 (numpy.ndarray): Original sequence data as an array of integer encodings.
    - Y0 (list of numpy.ndarray): Original label data as a list containing a single array of integer encodings.

    Returns:
    - numpy.ndarray: Reformatted sequence data.
    - list of numpy.ndarray: Reformatted label data, wrapped in a list.
    """
    # Calculate the number of data points needed
    #num_points = ceil_div(len(Y0[0]), SL)
    num_points = CL_max + SL
    # Initialize arrays to hold the reformatted data
    Xd = [np.zeros(num_points)]
    
    # Fill the initialized arrays with data in blocks
    for i in range(num_points):
        # Xd[i] = X0[SL * i : SL * (i + 1) + CL_max]
        # Yd[0][i] = Y0[0][SL * i : SL * (i + 1)]
        if i < len(X0):
            Xd[0][i] = X0[i]
        else: Xd[0][i] = float(np.random.randint(1,5))

    return np.asarray(Xd), [Y0]


def create_datapoints(seq, label):
    """
    This function first converts the sequence into an integer array, where
    A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    negative, then reverse complementing is done. The labels 
    are directly used as they are, converted into an array of integers,
    where 0, 1, 2 correspond to no splicing, acceptor, donor 
    respectively. It then calls reformat_data and one_hot_encode
    and returns X, Y which can be used by Pytorch Model.

    Parameters:
    - seq (str): The nucleotide sequence.
    - strand (str): The strand information ('+' or '-').
    - label (str): A string representation of labels for each nucleotide.

    Returns:
    - tuple: A tuple containing the one-hot encoded sequence and labels.
    """
    # No need to reverse complement the sequence, as sequence is already reverse complemented from previous step
    #seq = 'N' * (CL_max // 2) + seq + 'N' * (CL_max // 2)
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    # Convert label string to array of integers
    label_array = np.array(list(map(int, list(label))))
    X0 = np.asarray(list(map(int, list(seq))))

    Y0 = label_array
    Y0 = np.asarray(list(Y0))
    Xd, Yd = reformat_data(X0, Y0)
    X, Y = one_hot_encode(Xd, Yd)

    return X, Y