"""
train.py

Train a MiniGPT model
"""

import argparse
import torch
from torch.utils.data import DataLoader
import datasets
import transformers
from transformer_lens.utils import tokenize_and_concatenate
import numpy as np
import tqdm
import plotly.express as px
from model import MiniGPT

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
        self.losses = []
        self.accuracies = []


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
                self.losses.append(loss.item())
                progress_bar.update()
                progress_bar.set_description(f'Epoch {epoch+1}, loss: {loss:.3f},'
                                             f'accuracy: {accuracy:.2f}')
                if i >= self.steps_per_epoch:
                    break

            # validation
            correct_preds = torch.concat(
                [self.validation_step(batch) for batch in self.test_loader()])
            accuracy = correct_preds.float().mean().item()
            self.accuracies.append(accuracy)


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



def plot_loss_and_accuracy(losses, accuracies, n_epochs, dataset_name):
    """ Plot the losses and accuracies of the training process
    :param losses: list, loss value of each timestep
    :param accuracy: list, accuracy evaluated after each epoch
    :param n_epochs: int, number of training epochs
    :param dataset_name: str, prefix of the jpg files
    :return: None
    """

    # loss curve
    fig = px.line(y=losses, x=range(1, len(losses)+1))
    fig.update_layout(
        title=f'Training loss, {dataset_name} dataset<br><sup>{n_epochs} epochs</sup>',
        xaxis_title='Steps',
        yaxis_title='Loss',
        legend_title=None,
        font={'family': "Courier New, monospace", 'size': 18, 'color': "black"},
    )

    fig.write_image(f'{dataset_name}_loss_{n_epochs}.jpg')

    # accuracies
    fig = px.line(y=accuracies, x=range(1, n_epochs+1))
    fig.update_layout(
        title=f'Validation accuracy, {dataset_name} dataset<br><sup>{n_epochs} epochs</sup>',
        xaxis_title='Epochs',
        yaxis_title='Accuracy [%]',
        legend_title=None,
        font={'family': "Courier New, monospace", 'size': 18, 'color': "black"},
    )

    fig.write_image(f'{dataset_name}_accuracy_{n_epochs}.jpg')


def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to train on',
                        choices=['pile-10k', 'wikipedia-de'])
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', required=True)
    parser.add_argument('--steps', type=int, help='Number of steps per epoch', default=200)
    return parser.parse_args()

def get_dataset(dataset_name):
    """ Download the desired dataset
    :param dataset_name: str, name of the dataset to download
    :return: datasets.DatasetDict object, the downloaded dataset
    """

    if dataset_name == 'pile-10k':
        return datasets.load_dataset('NeelNanda/pile-10k', split='train').remove_columns('meta')
    elif dataset_name == 'wikipedia-de':
        return datasets.load_dataset('wikipedia', '20220301.de', split='train')
    return None


def main():
    """ Main function
        Get name of dataset from command-line args and download it
    :return: None
    """

    args = parse_args()

    # define training hyperparams
    batch_size = 16
    epochs = args.epochs
    max_steps_per_epoch = args.steps
    learning_rate= 1e-3
    weight_decay = 1e-2

    # load dataset
    ds = get_dataset(args.dataset)

    # get tokenizer from huggingface GPT2 implementation
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    # tokenize and concatenate dataset
    tokenized_dataset = tokenize_and_concatenate(ds, tokenizer, streaming=False, max_length=1024,
                                                 column_name="text", add_bos_token=True,
                                                 num_proc=4)

    # training
    # define model parameters
    n_layers = 6
    d_vocab = tokenizer.vocab_size
    context_length = 1024
    d_model = 256
    n_heads = 12
    d_head = 64

    device = torch.device('cuda')

    model = MiniGPT(n_layers, d_vocab, context_length, d_model, n_heads, d_head).to(device)
    trainer = TransformerTrainer(model, batch_size, epochs, max_steps_per_epoch, learning_rate,
                                 weight_decay, tokenized_dataset, device)
    trainer.train()

    # save model parameters
    torch.save(model.state_dict(), f'minigpt_{args.dataset}_{epochs}_epochs_weights')

    # plot loss and accuracy
    plot_loss_and_accuracy(trainer.losses, trainer.accuracies, epochs, f'{args.dataset}')

if __name__ == '__main__':
    main()
