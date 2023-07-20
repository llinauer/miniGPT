"""
download_data.py

Download the datasets used for training MiniGPT
"""

import argparse

def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['pile-10k', 'wikipedia'],
                        required=True)
    return parser.parse_args()
