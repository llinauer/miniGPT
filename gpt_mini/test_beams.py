"""
test_beams.py

Test the Beams class in generate.py
"""

import torch
import transformers
from model import MiniGPT
from generate import Beams
from transformer_lens import EasyTransformer


def test_beam_generation(model, tokenizer):

    beams = Beams(
        model,
        tokenizer,
        logprob_sums = torch.tensor([-10.0, -15.0, -20.0]),
        tokens = torch.tensor([
            [5661, 318, 262, 2368],
            [5661, 318, 262, 1218],
            [5661, 318, 262, 717],
        ])
    )

    print("Testing generate, without no_repeat_ngram_size argument:")
    new_beams = beams.generate(toks_per_beam=2)
    new_beams.print()
    assert new_beams.logprobs_and_completions[0][1] == "this is the third time"


def load_gpt2_weights(model, reference_gpt2):

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
            mapped_weights[key] = trained_weights[key]
        else:
            mapped_weights[transformer_weight_map[key]] = reference_gpt2.state_dict()[key]

    model.load_state_dict(mapped_weights, strict=False)
    return model


def main():

    reference_gpt2 = EasyTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False).to('cpu')
    tokenizer = reference_gpt2.tokenizer
    model = MiniGPT(12, reference_gpt2.tokenizer.vocab_size, 1024, 768, 12, 64)
    # load gpt2 weights
    model = load_gpt2_weights(model, reference_gpt2)
    

    test_beam_generation(model, tokenizer)

if __name__ == '__main__':
    main()
