"""
download_data.py

Download the datasets used for training MiniGPT
"""

import argparse
import datasets

def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of datset to download',
                        choices=['pile-10k', 'de-wikipedia'], required=True)
    parser.add_argument('-o', '--output_path', type=str,
                        help='Path to store dataset to. (Default = data)', default='data')
    return parser.parse_args()


def main():
    """ Main function
        Get name of dataset from command-line args and download it
    :return: None
    """

    args = parse_args()

    # load dataset
    if args.dataset == 'pile-10k':
        ds = datasets.load_dataset('NeelNanda/pile-10k')
    elif args.dataset == 'de-wikipedia':
        ds = datasets.load_dataset('wikipedia', '20220301.de')

    # save dataset to path
    ds.save_to_disk(args.output_path)


if __name__ == '__main__':
    main()
