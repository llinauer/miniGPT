"""
test_beams.py

Test the Beams class in generate.py
"""

import torch
import transformers
from model import MiniGPT
from generate import Beams


def test_beam_generation(beams):

    print("Testing generate, without no_repeat_ngram_size argument:")
    new_beams = beams.generate(toks_per_beam=2)
    new_beams.print()
    assert new_beams.logprobs_and_completions[0][1] == "this is the third time"



def main():

    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    model = MiniGPT(6, tokenizer.vocab_size, 1024, 256, 12, 64)
    
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

