"""
test_beams.py

Test the Beams class in generate.py
"""

import torch
from transformer_lens import EasyTransformer
from model import MiniGPT
from generate import Beams


def test_beam_generation(model, tokenizer):
    """ Test the beam generation method of Beams class
    :param model: nn.Module, transformer model
    :param tokenizer: transformers.tokenizer, tokenizer to use
    """

    # first test case
    beams = Beams( model, tokenizer,
        logprob_sums = torch.tensor([-10.0, -15.0, -20.0]),
        tokens = torch.tensor([
            [5661, 318, 262, 2368], # this is the third
            [5661, 318, 262, 1218], # this is the second
            [5661, 318, 262, 717], # this is the first
        ])
    )

    print("Testing generate, without no_repeat_ngram_size argument:")
    new_beams = beams.generate(toks_per_beam=2)
    new_beams.print()
    assert new_beams.logprobs_and_completions[0][1] == "this is the third time"

    # second test case
    beams = Beams(model, tokenizer,
        logprob_sums = torch.tensor([0.]),
        tokens = torch.tensor([
            [464, 1181,  286,  435, 8480,  318] # the state of alaska is
        ])
    )

    new_beams = beams.generate(toks_per_beam=15)
    new_beams.print()
    assert new_beams.logprobs_and_completions[9][1] == "The state of alaska is located"


def test_beam_filtering(model, tokenizer):
    """ Test the beam filtering method of Beams class
    :param model: nn.Module, transformer model
    :param tokenizer: transformers.tokenizer, tokenizer to use
    """

    logprob_sums = torch.tensor([-1.0, -2.0])
    tokens = torch.tensor([
        [19485, 13], # Stop .
        [19485, tokenizer.eos_token_id] # Stop EOS
    ])

    beams_with_eos = Beams(model, tokenizer, logprob_sums, tokens)
    best_beams, early_terminations = beams_with_eos.filter(2)

    torch.testing.assert_close(best_beams.logprob_sums, logprob_sums[[0]])
    torch.testing.assert_close(best_beams.tokens, tokens[[0]])

    assert early_terminations.logprobs_and_completions == [(-2.0, "Stop" + tokenizer.eos_token)]

    print("All tests for `filter` passed!")



def load_gpt2_weights(model, reference_gpt2):
    """ Load the fully trained gpt2 weights
    :param model: nn.Module, transformer model
    :param reference.gpt2: transformer_lense.HookedTransformer, fully trained gpt2 transformer
    :return: nn.Module, model with loaded weights
    """

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

    # map weights
    mapped_weights = {}
    for key in reference_gpt2.state_dict().keys():
        if not transformer_weight_map.get(key, None):
            mapped_weights[key] = reference_gpt2.state_dict()[key]
        else:
            mapped_weights[transformer_weight_map[key]] = reference_gpt2.state_dict()[key]

    model.load_state_dict(mapped_weights, strict=False)
    return model


def main():
    """ main function, load model, load gpt2 weights and run tests """
    reference_gpt2 = EasyTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False).to('cpu')
    tokenizer = reference_gpt2.tokenizer
    model = MiniGPT(12, reference_gpt2.tokenizer.vocab_size, 1024, 768, 12, 64)
    # load gpt2 weights
    model = load_gpt2_weights(model, reference_gpt2)

    # run tests
    test_beam_generation(model, tokenizer)
    test_beam_filtering(model, tokenizer)

if __name__ == '__main__':
    main()
