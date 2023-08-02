"""

test_model.py

Test if all the model layers output the correct shape
"""

import torch
from model import LayerNorm, Embedding, Unembedding, PositionalEmbedding, Attention, MLP
from model import TransformerBlock, MiniGPT
from transformer_lens import EasyTransformer


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


def trained_gpt2_test(layer, gpt2_layer, weight_map, layer_input):
    """ Load weights from a fully trained gpt2-style transformer layer
        and compare the outputs of it to the custom layer on some input tokens

    :param layer: nn.Module, Transformer layer to test
    :param gpt2_layer: nn.Module, Fully Trained GPT2-style layer
    :param weight_map: dict, map weight names from the trained layer to the custom layer
    :param layer_input: torch.tensor, input to the layer
    :return: None
    """
 
    # load weights from the trained layer
    trained_weights = gpt2_layer.state_dict()
    # map weights
    mapped_weights = {val: trained_weights[key] for key, val in weight_map.items()}

    layer.load_state_dict(mapped_weights, strict=False)
    output = layer(layer_input)
    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f'{comparison.sum()/comparison.numel():.2%} of the values are correct')

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
