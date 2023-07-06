"""
model.py

Definition of the transformer model

"""

import math
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


class Attention(nn.Module):
    """ Attention layer """

    def __init__(self, d_model, n_heads, d_head, init_std=0.02):
        """ Initialize an attention layer with n_heads attention heads, each with dimension d_head

        :param d_model: int, size of the transformer model
        :param n_heads: int, number of attention heads in the layer
        :param d_head: int, dimension of each attention head
        :param init_std: float, standard deviation for initializing the weights (default = 0.02)
        """
        super().__init__()

        # create Q,K & V weights and initialize them
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.normal_(self.W_Q, std=init_std)
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.normal_(self.W_K, std=init_std)
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        nn.init.normal_(self.W_V, std=init_std)

        # create Q, K & V biases
        self.b_Q = nn.Parameter(torch.zero_(n_heads, d_head))
        self.b_K = nn.Parameter(torch.zero_(n_heads, d_head))
        self.b_V = nn.Parameter(torch.zero_(n_heads, d_head))

        # W_O transforms back from d_head to d_model
        self.W_O = nn.Parameter(torch.empty(n_heads, d_head, d_model))
        nn.init.normal_(self.W_O, std=init_std)
        self.b_O = nn.Parameter(torch.zero_(n_heads, d_model))

        # add IGNORE buffer and set it to a small, non-zero number for masking
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))

    def apply_causal_mask(self, attention):
        """ Mask out non-causal pairs of source/target tokens in the attention.
            A non-causal token pair is one, where the target token has a position > than the source token

        :param attention: torch.tensor(batch, n_heads, query_position, key_position), attention pattern
        :return: torch.tensor(batch, n_heads, query_position, key_position), lower-triangular form of the attention
                                                                             pattern
        """

        mask = torch.triu(torch.ones(attention.size(-2), attention.size(-1), device=attention.device),
                          diagonal=1).bool()
        attention.masked_fill_(mask, self.IGNORE)
        return attention

    def forward(self, inputs):
        """ Each attention head calculates the so-called scaled dot-product attention:
                Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) * V
            (see: https://arxiv.org/pdf/1706.03762.pdf)

        In words, this means that for each head in the attention layer, we do the following:
            -) Transform the inputs to queries, keys and values with W_Q, W_K and W_V
            -) Multiply every pair of queries and keys and softmax the result -> attention pattern
            -) Multiply the attention pattern with the W_V matrix
            -) Transform from d_head dimensions back to d_model dimensions with W_O

        In the end, the outputs of all heads are added together

        :param inputs: torch.tensor(batch_size, position, d_model), layer inputs
        :return: torch.tensor(batch, position, d_model), sum of scaled dot-product attentions of all heads
        """

        # transform inputs to queries and keys, do this for every head at the same time
        queries = einops.einsum(inputs, self.W_Q, 'batch position d_model, n_heads d_model d_head -> '
                                                  'batch position n_heads d_head')
        queries += self.b_Q

        keys = einops.einsum(inputs, self.W_K, 'batch position d_model, n_heads d_model d_head -> '
                                               'batch position n_heads d_head')
        keys += self.b_K

        # multiply queries with keys pairwise, this creates an attention patten for each pair of tokens
        # the queries correspond to the source tokens, the keys to the target tokens
        attention = einops.einsum(queries, keys, 'batch query_position n_heads d_head, '
                                                 'batch key_position n_heads, d_head '
                                                 '-> batch n_heads query_position key_position')

        # scale the attention by dividing it by sqrt(d_head)
        attention /= math.sqrt(self.b_Q.shape[1])

        # attention is now of shape (batch n_heads, query_position, key_position) (the query_position and
        # key_position dimensions are the same), this means, for each head in every element of the batch, we have
        # a query_position x key_position matrix
        # Each element of this matrix corresponds to the pairwise attention between two tokens, one source and one
        # target token. However, we want the attention to only be calculated between a target token, that is
        # at most in the same position as the source token. Otherwise the model would "look into the future"
        # -> make the attention matrix lower-triangular
        attention = self.apply_causal_mask(attention)

        # finally, apply softmax along the query_position dimension to get one probability distribution per
        # source token, over all target tokens
        attention = attention.softmax(dim=-1)

        # transform inputs to values
        values = einops.einsum(inputs, self.W_V, 'batch position d_model, n_heads d_model d_head -> '
                                                 'batch position n_heads d_head')
        values += self.b_V

        # multiply attention with values
        # attention is of shape (batch, n_heads, query_position, key_position)
        # values is of shape (batch, position, n_heads, d_head)
        out = einops.einsum(attention, values, 'batch n_heads query_position key_position, '
                                               'batch key_position n_heads d_head '
                                               '-> batch query_position n_heads d_head')

        # transform to d_model dimension and sum over all heads
        out += einops.einsum(out, self.W_O, 'batch position n_heads d_head, n_heads d_head d_model '
                                            '-> batch position d_model')
        out += self.b_O

        return out


class MLP(nn.Module):
    """ MLP layer """

    def __init__(self, d_model, init_std=0.02):
        """ Initialize an MLP layer with one hidden layer with 4*d_model neuros
        :param d_model: int, size of the transformer model
        :param init_std: float, standard deviation for initializing the weights (default = 0.02)
        """
        
        super().__init__()

        # create weights and biases for hidden and output layer
        self.W_hidden = nn.Parameter(torch.empty(d_model, 4*d_model))
        torch.nn.init.normal_(self.W_hidden, std=init_std)
        self.b_in = nn.Parameter(torch.zeros(4*d_model))

        self.W_out = nn.Parameter(torch.empty(4*d_model, d_model))
        torch.nn.init.normal_(self.W_out, std=init_std)
        self.b_out = nn.Parameter(torch.zeros(d_model))
    
        # activation
        self.activation = nn.GELU()

    def forward(self, inputs):
        """ 
        :param inputs: torch.tensor(batch_size, position, d_model), layer inputs
        :return: torch.tensor(batch, position, d_model), output of MLP layer
        """

        # process hidden layer
        out = einops.einsum(inputs, self.W_hidden, 'batch position d_model, d_model d_hidden 
                                                    -> batch position d_hidden')
        out += self.b_hidden
        # activation
        out = self.activation(out)

        # process output layer
        out = einops.einsum(out, self.W_out, 'batch position d_hidden, d_hidden d_model 
                                              -> batch position d_model')
        out += self.b_out

        return out


class Unembed(nn.Module):
    """ Unenbedding, transform to logits """

    def __init__(self, d_model, d_vocab, init_std):
        """ Initialize the unembedding layer with W_U and b_U

        :param d_model: int, size of the transformer model
        :param d_vocab: int, size of the vocabulary
        :param init_std: float, standard deviation for initializing the weights
        """

        super().__init__()

        self.W_U = nn.Parameter(torch.empty(d_model, d_vocab))
        torch.nn.init.normal_(self.W_U, std=init_std)
        self.b_U = nn.Parameter(torch.zeros(d_vocab))

