import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    
    # Train parameters
    parser.add_argument(
        '--batch_size', default=4, type=int)
    parser.add_argument(
        '--seed', default=0, type=int)
    parser.add_argument(
        '--se_train_file', default='./data/sevir/sevir_train_r.npy')
    parser.add_argument(
        '--se_test_file', default='./data/sevir/sevir_test_r.npy')
    parser.add_argument(
        '--mmse_train_file', default='./data/sevir/sevir_train_m.npy')
    parser.add_argument(
        '--mmse_test_file', default='./data/sevir/sevir_test_m.npy')
    parser.add_argument(
        '--log_file', default='./save/logger.txt')
    parser.add_argument(
        '--num_workers', default=8, type=int)
    parser.add_argument(
        '--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument(
        '--lr_betas', default=(0.9, 0.999), type=float, help='Learning betas')
    parser.add_argument(
        '--vil_thresholds', default=[0.14, 0.70, 3.50, 6.90], type=float, nargs='*')

    return parser
