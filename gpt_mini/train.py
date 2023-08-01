"""
train.py

Train a MiniGPT model
"""

import argparse
import torch
import datasets
import transformers
from transformer_lens.utils import tokenize_and_concatenate


class TransformerTrainer:
    """ class for training the MiniGPT transformer """

    def __init__(self, model, bs, n_epochs, steps_per_epoch, lr, wd):
        """ Constructor

        :param model: nn.Module, transformer model to train
        :param bs: int, batch size
        :param n_epochs: int, number of epochs to train
        :param steps_per_epochs: int, number of training steps per epoch
        :param lr: float, learning rate
        :param wd: float, weight decay rate
        """

        super().__init__()
        self.model = model
        self.bs = bs
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.wd = wd
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.step = 0





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




