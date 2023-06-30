"""
model.py

Definition of the transformer model

"""

import torch
import torch.nn as nn
import einops


class MiniGPT(nn.Module):
    """ GPT-style transformer model """

    def __init__(self):
        pass


class LayerNorm(nn.Module):
    """ LayerNorm:
        Normalize inputs to a layer over all the neurons of that layer, then
        do an affine transformation with learnable parameters gamma and beta"""

    def __init__(self, d_model, epsilon=1e-5):
        """ Initialize the LayerNorm layer. Create the affine transformation parameters
            gamma (scaling) and beta (translation).

        :param d_model: int, size of the transformer model
        :param epsilon: float, added to the denominator in the normalization for numerical stability
        """

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, inputs):
        """ Normalize the input over all neurons in the layer. Then apply scaling with gamma and translate
            with beta.

        :param inputs: torch.tensor(batch_size, position, d_model), layer inputs
        :return: torch.tensor(batch_size, position, d_model), layer normalization of inputs
        """

        # calculate the mean and variance over all neurons in the layer ( = d_model dimension)
        mean = einops.reduce(inputs, 'batch position d_model -> batch position 1', 'mean')
        var = einops.reduce(torch.pow(inputs - mean, 2), 'batch position d_model -> batch position 1', 'mean')

        # subtract the mean, divide by sqrt(var) and transform
        normalized_inputs = (inputs - mean) / torch.sqrt(var + self.epsilon)
        return (normalized_inputs * self.gamma) + self.beta


class Embedding(nn.Module):
    """ Embedding:
        Translate the input tokens to the embedding space
    """

    def __init__(self, d_model, d_vocab, init_std):
        """ Initialize the Embedding layer with the embedding matrix

        :param d_model: int, size of the transformer model
        :param d_vocab: int, size of the vocabulary
        :param init_std: float, standard deviation for initializing the weights
        """

        self.W_E = nn.Parameter(torch.empty((d_vocab, d_model)))
        nn.init.normal_(self.W_E, std=init_std)

    def forward(self, tokens):
        """ Embed the input tokens. The input tokens are integers; we want to select for each token
            the row of the W_E matrix with that index.

        :param tokens: torch.tensor(batch, position d_vocab), input tokens
        :return: torch.tensor(batch, position d_model), embeddings
        """

        embeddings = self.W_E[tokens]
        return embeddings


class PositionalEmbedding(nn.Module):
    """ Positional Embedding:
    """

    def __init__(self, context_length, d_model, init_std=0.02):
        """ Initialize the positional embedding layer with the W_pos matrix

        :param context_length: int, context length of the transformer
        :param d_model: int, size of the transformer model
        :param init_std: float, standard deviation for initializing the weights
        """
        super().__init__()
        self.W_pos = nn.Parameter(torch.empty((context_length, d_model)))
        nn.init.normal_(self.W_pos, std=init_std)

    def forward(self, tokens):
        """ Take the first n rows from W_pos (where n = number of tokens in sequence) for every sequence in a
            batch and return it.
            The i-th row in W_pos corresponds to the positional embedding of the i-th token in the sequence

        :param tokens: torch.tensor(batch, position, d_model), input tokens
        :return: torch.tensor(batch, position, d_model), positional embeddings
        """

        # the positional embeddings should have the same shape as the embeddings, since we are going
        # to add them together -> create a tensor of the same shape

        # take the first n rows from W_pos, where n is the number of tokens in the sequence
        pos_embed = self.W_pos[:tokens.size(1)] # shape = (position, d_model)
        # repeat for every batch -> create a tensor of shape (batch, position, d_model)
        pos_embed = einops.repeat(pos_embed, 'position d_model -> batch position d_model', batch=tokens.size(0))
        return pos_embed
