"""
test.py 

Test the transformer model by letting it generate text
"""

import argparse
import torch
import transformers


def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Path to the weights file', required=True)
    parser.add_argument('--decoding-method', type=str, choices=['greedy', 'beam', 'sample'],
                        help='Method to decode the transformer output')
    return parser.parse_args()
