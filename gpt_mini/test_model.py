"""

test_model.py

Test if all the model layers output the correct shape
"""

import torch
from model import LayerNorm, Embedding, Unembedding, PositionalEmbedding, Attention, MLP   
from model import TransformerBlock, MiniGPT


def rand_float_test(layer, input_shape, expected_shape):
    """ Pass random floats through the layer and check the shape
    :param layer: nn.Module, Transformer layer to test
    :param input_shape: tuple, shape of the layer input
    :param expected_shape: tuple, expected shape of the layer output
    """

    random_input = torch.randn(input_shape)
    output = layer(random_input)
    if tuple(output.shape) == expected_shape:
        print('OK!')
    else:
        print(f'FAIL! Output shape {tuple(output.shape)} != {expected_shape}')

def rand_int_test(layer, input_shape, expected_shape):
    """ Pass random integers through the layer and check the shape
    :param layer: nn.Module, Transformer layer to test
    :param input_shape: tuple, shape of the layer input
    :param expected_shape: tuple, expected shape of the layer output
    """
    random_input = torch.randint(100, 1000, input_shape)
    output = layer(random_input)
    if tuple(output.shape) == expected_shape:
        print('OK!')
    else:
        print(f'FAIL! Output shape {tuple(output.shape)} != {expected_shape}')


def test_model():

    print('Test LayerNorm layer')
    rand_float_test(LayerNorm(768), (2, 4, 768), (2, 4, 768))
    
    print('Test Embedding layer')
    rand_int_test(Embedding(768, 4000), (2, 4), (2, 4, 768))
    print('Test Positional embedding layer')
    rand_int_test(PositionalEmbedding(1000, 768), (2, 120), (2, 120, 768))
    print('Test Unembedding layer')
    rand_float_test(Unembedding(768, 2000), (2, 4, 768), (2, 4, 2000))
    print('Test Attention layer')
    rand_float_test(Attention(768, 6, 64), (2, 4, 768), (2, 4, 768))
    print('Test MLP layer')
    rand_float_test(MLP(768), (2, 4, 768), (2, 4, 768))
    print('Test TransformerBlock')
    rand_float_test(TransformerBlock(768, 6, 64), (2, 4, 768), (2, 4, 768))
    print('Test Full Transformer')
    rand_int_test(MiniGPT(6, 2000, 1000, 768, 6, 64), (2, 4), (2, 4, 2000))


if __name__ == '__main__':
    test_model()
