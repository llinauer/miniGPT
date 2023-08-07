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
    mapped_weights = {}
    for key in trained_weights.keys():
        if not weight_map.get(key, None):
            mapped_weights[key] = trained_weights[key]
        else:
            mapped_weights[weight_map[key]] = trained_weights[key]

    layer.load_state_dict(mapped_weights, strict=False)
    output = layer(layer_input)
    try: reference_output = gpt2_layer(layer_input)
    except: reference_output = gpt2_layer(layer_input, layer_input, layer_input)
    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f'{comparison.sum()/comparison.numel():.2%} of the values are correct')

def test_model_shapes():
    print('Testing output shapes of MiniGPT model')
    print('-----')
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
    print('-----')

def test_model_outputs():
    print('Testing output values of MiniGPT model')
    print('-----')

    # input string
    input_str = 'If you can read this, you are too close'

    # load the trained gpt2-style model
    reference_gpt2 = EasyTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False).to('cpu')

    # tokenize the input
    input_tokens = reference_gpt2.to_tokens(input_str)
    # pass the tokens through the trained model
    logits, cache = reference_gpt2.run_with_cache(input_tokens)

    print('Test LayerNorm layer')
    trained_gpt2_test(LayerNorm(768), reference_gpt2.ln_final, {'w': 'gamma', 'b': 'beta'},
                      cache['resid_post', 11])

    print('Test Embedding layer')
    trained_gpt2_test(Embedding(768, reference_gpt2.tokenizer.vocab_size), reference_gpt2.embed,
                      {}, input_tokens)

    print('Test Positional embedding layer')
    trained_gpt2_test(PositionalEmbedding(1024, 768), reference_gpt2.pos_embed, {}, input_tokens)

    print('Test Unembedding layer')
    trained_gpt2_test(Unembedding(768, reference_gpt2.tokenizer.vocab_size),
                      reference_gpt2.unembed, {}, cache['ln_final.hook_normalized'])
    print('Test Attention layer')
    trained_gpt2_test(Attention(768, 12, 64), reference_gpt2.blocks[0].attn,
                      {}, cache['normalized', 0, 'ln1'])
    print('Test MLP layer')
    trained_gpt2_test(MLP(768), reference_gpt2.blocks[0].mlp,
                      {'W_in': 'W_hidden', 'b_in': 'b_hidden'}, cache['normalized', 0, 'ln2'])
    print('Test TransformerBlock')
    tb_weight_map = {'ln1.w': 'attention_ln.gamma', 'ln1.b': 'attention_ln.beta',
                     'ln2.w': 'mlp_ln.gamma', 'ln2.b': 'mlp_ln.beta',
                     'attn.W_Q': 'attention_layer.W_Q', 'attn.W_K': 'attention_layer.W_K',
                     'attn.W_V': 'attention_layer.W_V', 'attn.W_O': 'attention_layer.W_O',
                     'attn.b_Q': 'attention_layer.b_Q', 'attn.b_K': 'attention_layer.b_K',
                     'attn.b_V': 'attention_layer.b_V', 'attn.b_O': 'attention_layer.b_O',
                     'mlp.W_in': 'mlp.W_hidden', 'mlp.b_in': 'mlp.b_hidden'}
    trained_gpt2_test(TransformerBlock(768, 12, 64), reference_gpt2.blocks[0], tb_weight_map,
                      cache['resid_pre', 0])

    print('Test Full Transformer')
    # map all weight names from reference_gpt2 to MiniGPT
    transformer_weight_map = {}
    for weight_name in reference_gpt2.state_dict().keys():

        new_weight = weight_name

        # ln1 -> attention_ln
        new_weight = new_weight.replace('ln1.w', 'attention_ln.gamma')
        new_weight = new_weight.replace('ln1.b', 'attention_ln.beta')
        # ln2 -> mlp_ln
        new_weight = new_weight.replace('ln2.w', 'mlp_ln.gamma')
        new_weight = new_weight.replace('ln2.b', 'mlp_ln.beta')
        # ln_final.w, ln_final.b -> ln_final.gamma, ln_final.beta
        new_weight = new_weight.replace('ln_final.w', 'ln_final.gamma')
        new_weight = new_weight.replace('ln_final.b', 'ln_final.beta')
        # attn -> attention_layer
        new_weight = new_weight.replace('attn', 'attention_layer')
        # W_in, b_in -> W_hidden, b_hidden
        new_weight = new_weight.replace('W_in', 'W_hidden').replace('b_in', 'b_hidden')

        transformer_weight_map[weight_name] = new_weight

    trained_gpt2_test(MiniGPT(12, reference_gpt2.tokenizer.vocab_size, 1024, 768, 12, 64),
                      reference_gpt2, transformer_weight_map, input_tokens)


if __name__ == '__main__':
    test_model_shapes()
    test_model_outputs()
