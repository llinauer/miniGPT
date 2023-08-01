"""
train.py

Train a MiniGPT model
"""

import argparse
import torch
import torch.nn.functional as F
import datasets
import transformers
from transformer_lens.utils import tokenize_and_concatenate


def get_log_probs(tokens, logits):
    """ Calculate the log probabilities for each token

    :param tokens: torch.tensor(batch_size, position), input tokens
    :param logits: torch.tensor(batch_size, position, d_vocab), output of transformer model
    :return: torch.tensor(batch_size, position), log probabilities for each token
    """

    # calculate log probs of the logits
    log_probs = logits.log_softmax(dim=-1)

    # for each token (until the second to last), get the log prob of the next token
    # -> store this in log_probs_for_tokens
    log_probs_for_tokens = log_probs[:, :-1].gather(
        dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens


class TransformerTrainer:
    """ class for training the MiniGPT transformer """

    def __init__(self, model, bs, n_epochs, steps_per_epoch, lr, wd, device='cpu'):
        """ Constructor

        :param model: nn.Module, transformer model to train
        :param bs: int, batch size
        :param n_epochs: int, number of epochs to train
        :param steps_per_epochs: int, number of training steps per epoch
        :param lr: float, learning rate
        :param wd: float, weight decay rate
        :param device: str or torch.device obj, device to send tensors to
        """

        super().__init__()
        self.model = model
        self.bs = bs
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.wd = wd
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.device = device
        self.step = 0


    def training_step(self, batch):
        """ Do one training step with the batch

        :param batch: dict, dictionary containing a batch of the tokenized dataset
        :return: float, loss value
        """

        tokens = batch['tokens'].to(self.device)
        logits = self.model(tokens)
        loss = -get_log_probs




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




