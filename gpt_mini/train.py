"""
train.py

Train a MiniGPT model
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
import transformers
from transformer_lens.utils import tokenize_and_concatenate
from model import MiniGPT
import numpy as np
import tqdm

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

    def __init__(self, model, bs, n_epochs, steps_per_epoch, lr, wd, dataset, device='cpu'):
        """ Constructor

        :param model: nn.Module, transformer model to train
        :param bs: int, batch size
        :param n_epochs: int, number of epochs to train
        :param steps_per_epochs: int, number of training steps per epoch
        :param lr: float, learning rate
        :param wd: float, weight decay rate
        :param dataset: datasets.Dataset, dataset to use
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
        dataset_dict = dataset.train_test_split(test_size=1000)
        self.train_ds = dataset_dict['train']
        self.test_ds = dataset_dict['test']
        self.step = 0


    def training_step(self, batch):
        """ Do one training step with the batch

        :param batch: dict, dictionary containing a batch of the tokenized dataset
        :return: float, loss value
        """

        tokens = batch['tokens'].to(self.device)
        logits = self.model(tokens)
        loss = -get_log_probs(tokens, logits).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1

        return loss


    def validation_step(self, batch):
        """ Do one validation step with the batch

        :param batch: dict, dictionary containing a batch of the tokenized dataset
        :return: torch.tensor(batch_size, position), boolean tensor, True if prediction is correct
        """

        tokens = batch['tokens'].to(self.device)
        logits = self.model(tokens)[:, :-1]
        predicted_tokens = logits.argmax(dim=-1)
        correct_predictions = (predicted_tokens == tokens[:, 1:]).flatten()

        return correct_predictions


    def train(self):
        """ Training loop """

        accuracy = np.nan
        progress_bar = tqdm.tqdm(total = self.steps_per_epoch * self.n_epochs)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(self.train_loader()):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f'Epoch {epoch+1}, loss: {loss:.3f},'
                                             f'accuracy: {accuracy:.2f}')
                if i >= self.steps_per_epoch:
                    break

            # validation
            correct_preds = torch.concat(
                [self.validation_step(batch) for batch in self.test_loader()])
            accuracy = correct_preds.float().mean().item()


    def train_loader(self):
        """ Return the training dataloader
        :return: torch.utils.data.DataLoader, training dataloader
        """

        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True,
                          num_workers=4, pin_memory=True)

    def test_loader(self):
        """ Return the test dataloader
        :return: torch.utils.data.DataLoader, test dataloader
        """

        return DataLoader(self.test_ds, batch_size=self.bs, shuffle=True,
                          num_workers=4, pin_memory=True)




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
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 200
    learning_rate= 1e-3
    weight_decay = 1e-2

    # load dataset
    ds = datasets.load_dataset('NeelNanda/pile-10k', split='train').remove_columns('meta')

    # get tokenizer from huggingface GPT2 implementation
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    # tokenize and concatenate dataset
    tokenized_dataset = tokenize_and_concatenate(ds, tokenizer, streaming=False, max_length=1024,
                                                 column_name="text", add_bos_token=True,
                                                 num_proc=4)

    # training
    # define model parameters
    n_layers = 8
    d_vocab = tokenizer.vocab_size
    context_length = 1024
    d_model = 256
    n_heads = 6
    d_head = 64

    device = torch.device('cuda')

    model = MiniGPT(n_layers, d_vocab, context_length, d_model, n_heads, d_head).to(device)
    trainer = TransformerTrainer(model, batch_size, epochs, max_steps_per_epoch, learning_rate,
                                 weight_decay, tokenized_dataset, device)
    trainer.train()

if __name__ == '__main__':
    main()