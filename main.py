import os
import torch
import random
import argparse
import numpy as np
import time
import cProfile

from model.STAR import STARFramework
from tools.Trainer import Trainer, NewTrainer
from tools import DataSet

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='diginetica', help='dataset name: diginetica/yoochoose')
parser.add_argument('--fraction', default=4, help='1/4/64')
parser.add_argument('--validation', type=bool, default=False, help='use validation or not')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--do_pretrain', type=bool, default=True, help='item embedding dim')
parser.add_argument('--iter', type=int, default=100, help='number of epochs for glove')
parser.add_argument('--theta_coef', type=int, default=2.0, help='coefficient to control the time interval between actions')

parser.add_argument('--embedding_dim', type=int, default=180, help='item embedding dim')
parser.add_argument('--hidden_size', type=int, default=180, help='hidden state size')

parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay')
parser.add_argument('--w_dc', type=int, default=1e-5, help='weight decay rate')
parser.add_argument('--top_k', type=int, default=200, help='k in metrics')
parser.add_argument('--seed_value', type=int, default=8, help='seed')
parser.add_argument('--epsilon', type=float, default=10, help='epsilon value for Gaussian noise')

# Debugging-related arguments
parser.add_argument('--log_interval', type=int, default=10, help='Log every n batches')
parser.add_argument('--max_batches', type=int, default=None, help='Maximum number of batches to process for debugging')

args = parser.parse_args()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed_value)
    torch.cuda.manual_seed(args.seed_value)
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    torch.cuda.empty_cache()

    # Set device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories for dataset and results
    data_dir = os.path.join("Data", "cleaned", args.data_name)
    result_dir = os.path.join('Results', args.data_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Define the number of items based on the dataset
    if args.data_name == 'diginetica':
        n_items = 43097
    elif args.data_name == 'yoochoose':
        n_items = 37483

    # Create datasets
    train_dataset, test_dataset = DataSet.create_sets(data_dir, args, n_items)

    # Create the STAR framework (model)
    star_framework = STARFramework(device, n_items, args)
    star_framework.set_optimizer(args)

    # Initialize the trainer
    # Use NewTrainer or Trainer depending on your data
    if isinstance(train_dataset, dict):  # Assuming dataset is in dictionary format for NewTrainer
        trainer = NewTrainer(train_dataset, star_framework, star_framework.optimizer, args)
    else:
        trainer = Trainer(device)  # Use the original Trainer if dataset isn't in dictionary format

    # Training and saving model
    trainer.train(star_framework, train_dataset, test_dataset, args, result_dir)

    # Save the model checkpoint
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    checkPoints = {'model': star_framework.state_dict()}
    save_path = os.path.join(result_dir, "Star_model.pth")
    torch.save(checkPoints, save_path)
