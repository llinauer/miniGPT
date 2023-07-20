"""
train.py

Train a MiniGPT model
"""

import argparse
import datasets


def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, help='Path to a downloaded dataset')
    return parser.parse_args()


def main():
    """ Main function
        Get name of dataset from command-line args and download it
    :return: None
    """

    args = parse_args()

    # load dataset
    
