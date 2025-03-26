
import argparse
import os, sys, time
import random
import h5py
import numpy as np
import header
from create_data import create_data
from train import train
from test import test
# from openspliceai.calibrate import calibrate
# from openspliceai.transfer import transfer
# from openspliceai.predict import predict
# from openspliceai.variant import variant

__VERSION__ = header.__version__

def parse_args_create_data(subparsers):
    parser_create_data = subparsers.add_parser('create-data', help='Build dataset from data')
    parser_create_data.add_argument('--data', '-X', type=str, required=True, help='Input data file')
    parser_create_data.add_argument('--labels', '-y', type=str, required=True, help='Input labels file')
    # parser_create_data.add_argument('--annotation-gff', type=str, required=True, help='Path to the GFF file')
    # parser_create_data.add_argument('--genome-fasta', type=str, required=True, help='Path to the FASTA file')
    parser_create_data.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    # parser_create_data.add_argument('--parse-type', type=str, default='canonical', choices=['canonical', 'all_isoforms'], help='Type of transcript processing')
    # parser_create_data.add_argument('--biotype', type=str, default='protein-coding', choices=['protein-coding', 'non-coding'], help='Biotype of transcript processing')
    parser_create_data.add_argument('--chr-split', '-s', type=str, choices=['train-test','test'], default='train-test', help='Whether to obtain testing or both training and testing groups')
    '''AM: newly added flags below vv'''
    # parser_create_data.add_argument('--split-method', type=str, choices=['random', 'human'], default='random', help='Chromosome split method for training and testing dataset')
    # parser_create_data.add_argument('--split-ratio', type=float, default=0.8, help='Ratio of training and testing dataset')
    # parser_create_data.add_argument('--canonical-only', action='store_true', default=False, help='Flag to obtain only canonical splice site pairs')
    # parser_create_data.add_argument('--flanking-size', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    # parser_create_data.add_argument('--verify-h5', action='store_true', default=False, help='Verify the generated HDF5 file(s)')
    # parser_create_data.add_argument('--remove-paralogs', action='store_true', default=False, help='Remove paralogous sequences between training and testing dataset')
    # parser_create_data.add_argument('--min-identity', type=float, default=0.8, help='Minimum minimap2 alignment identity for paralog removal between training and testing dataset')
    # parser_create_data.add_argument('--min-coverage', type=float, default=0.5, help='Minimum minimap2 alignment coverage for paralog removal between training and testing dataset')
    # parser_create_data.add_argument('--write-fasta', action='store_true', default=False, help='Flag to write out sequences into fasta files')

def parse_args_train(subparsers):
    parser_train = subparsers.add_parser('train', help='Train the Thistle model')
    parser_train.add_argument('--epochs', '-n', type=int, default=10, help='Number of epochs for training')
    parser_train.add_argument('--scheduler', '-s', type=str, default="MultiStepLR", choices=["MultiStepLR", "CosineAnnealingWarmRestarts"], help="Learning rate scheduler")
    parser_train.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_train.add_argument("--patience", '-P', type=int, default=2, help="Number of epochs to wait before early stopping")
    parser_train.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_train.add_argument('--project-name', '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_train.add_argument('--exp-num', '-e', type=str, default=0, help="Experiment number")
    parser_train.add_argument('--random-seed', '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_train.add_argument('--train-dataset', '-train', type=str, required=True, help="Path to the training dataset")
    parser_train.add_argument('--test-dataset', '-test', type=str, required=True, help="Path to the testing dataset")
    parser_train.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for training")
    parser_train.add_argument('--model', '-m', default="thistle", type=str)

def parse_args_test(subparsers):
    parser_test = subparsers.add_parser('test', help='Test the Thistle model')
    parser_test.add_argument("--pretrained-model", '-m', type=str, required=True, help="Path to the pre-trained model")
    parser_test.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_test.add_argument('--project-name', '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_test.add_argument('--exp-num', '-e', type=str, default=0, help="Experiment number")
    parser_test.add_argument('--flanking-size', '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_test.add_argument('--random-seed', '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_test.add_argument('--test-dataset', '-test', type=str, required=True, help="Path to the testing dataset")
    parser_test.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for training")
    parser_test.add_argument('--log-dir', '-L', default="TEST_LOG", type=str)

def parse_args(arglist):
    parser = argparse.ArgumentParser(description='Translation start site prediction :)')
    # Create a parent subparser to house the common subcommands.
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create-data, train, test, calibrate, predict, transfer, variant')
    parse_args_create_data(subparsers)
    parse_args_train(subparsers)
    parse_args_test(subparsers)
    # parse_args_calibrate(subparsers)
    # parse_args_transfer(subparsers)
    # parse_args_predict(subparsers)
    # parse_args_variant(subparsers)
    if arglist is not None:
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    return args


def main(arglist=None):
    # ANSI Shadow
    # https://patorjk.com/software/taag/
    banner = '''
====================================================================
Deep learning framework to detect translation initiation sites
====================================================================                

 ___, ,  ___,_, ___,    _, 
' | |_|,' | (_,' | |   /_, 
  |'| |  _|_,_)  |'|__'\_  
  ' ' ` '   '    '   '   ` 
                                                                        

▄▄▄█████▓ ██░ ██  ██▓  ██████ ▄▄▄█████▓ ██▓    ▓█████ 
▓  ██▒ ▓▒▓██░ ██▒▓██▒▒██    ▒ ▓  ██▒ ▓▒▓██▒    ▓█   ▀ 
▒ ▓██░ ▒░▒██▀▀██░▒██▒░ ▓██▄   ▒ ▓██░ ▒░▒██░    ▒███   
░ ▓██▓ ░ ░▓█ ░██ ░██░  ▒   ██▒░ ▓██▓ ░ ▒██░    ▒▓█  ▄ 
  ▒██▒ ░ ░▓█▒░██▓░██░▒██████▒▒  ▒██▒ ░ ░██████▒░▒████▒
  ▒ ░░    ▒ ░░▒░▒░▓  ▒ ▒▓▒ ▒ ░  ▒ ░░   ░ ▒░▓  ░░░ ▒░ ░
    ░     ▒ ░▒░ ░ ▒ ░░ ░▒  ░ ░    ░    ░ ░ ▒  ░ ░ ░  ░
  ░       ░  ░░ ░ ▒ ░░  ░  ░    ░        ░ ░      ░   
          ░  ░  ░ ░        ░               ░  ░   ░  ░
                                                                         
    '''
    # print(banner, file=sys.stderr)
    print(banner)
    print(f"{__VERSION__}\n", file=sys.stderr)
    args = parse_args(arglist)
    
    if args.command == 'create-data':
        create_data.create_datafile(args)
        create_data.create_dataset(args)
    if args.command == 'train':
        train.train(args)
    elif args.command == 'test':
        test.test(args)
    # elif args.command == 'calibrate':
    #     calibrate.calibrate(args)
    # elif args.command == 'transfer':
    #     transfer.transfer(args)
    # elif args.command == 'predict':
    #     predict.predict_cli(args)
    # elif args.command == 'variant':
    #     variant.variant(args)

if __name__ == "__main__":
    main()