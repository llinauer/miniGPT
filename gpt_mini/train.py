"""
train.py

Train a MiniGPT model
"""

import argparse
import datasets
import transformers
from transformer_lens.utils import tokenize_and_concatenate


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

    # define training hyperparams
    batch_size = 64
    epochs = 10
    max_steps_per_epoch = 200
    learning_rate= 1e-3
    weight_decay = 1e-2

    # load dataset
    ds = datasets.load_dataset('NeelNanda/pile-10k', split='train').remove_columns('meta')

    # get tokenizer from huggingface GPT2 implementation
    tokenizer = tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    # tokenize and concatenate dataset
    tokenized_dataset = tokenize_and_concatenate(ds, tokenizer, streaming=False, max_length=768,
                                                 column_name="text", add_bos_token=True,
                                                 num_proc=4)

    # train/test split
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    train_loader = DataLoader(dataset_dict["train"], batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_dict["test"], batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)




