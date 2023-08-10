"""
test.py 

Test the transformer model by letting it generate text
"""

import argparse
import torch
import transformers
from pathlib import Path
import einops
from tqdm import tqdm
from model import MiniGPT

class Beams:
    """Class to store beams during beam search."""
    
    def __init__(self, model, tokenizer, logprob_sums, tokens):
        self.model = model
        self.tokenizer = tokenizer
        self. logprob_sums = logprob_sums
        self.tokens = tokens


    def new_beams(self, logprob_sums, tokens):
        """Creates a new Beams object with the same model and tokenizer."""
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx):
        """Allows you to take a slice of the beams object along the batch dimension."""
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self):
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]


    def generate(self, toks_per_beam, no_repeat_ngram_size=None):
        """
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with
        a repeating n-gram of this length.
        """

        # get logits and logprobs
        logits = self.model(self.tokens)[:, -1, :]
        logprobs = logits.log_softmax(dim=-1)

        # get the top k logprobs and their indices
        topk_logprobs, topk_token_idcs = logprobs.topk(k=toks_per_beam)

        logprob_sums_unpacked = einops.repeat(self.logprob_sums, "batch -> batch k",
                                              k=toks_per_beam).flatten()
        topk_logprobs_flattened = einops.rearrange(topk_logprobs, "batch k -> (batch k)")
 

        new_logprob_sums = sum([logprob_sums_unpacked, topk_logprobs_flattened])

        new_tokens = torch.concat([einops.repeat(self.tokens, "batch seq -> (batch k) seq",
                                                 k=toks_per_beam),
                                   einops.rearrange(topk_token_idcs, "batch k -> (batch k) 1")
                                  ], dim=-1)

        return self.new_beams(new_logprob_sums, new_tokens)



    def filter(self, num_beams):
        """
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `num_beams` which are also not terminated.

            early_terminations: Beams
                filtered version of self, containing all best `num_beams` which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        """

        # Get the indices of top `num_beams` beams
        top_beam_indices = self.logprob_sums.topk(k=num_beams, dim=0).indices.tolist()
        # Get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = torch.nonzero(new_tokens == self.tokenizer.eos_token_id)
    
        # Get the indices of the `num_beams` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]
    
        # Return the beam objects from these indices
        best_beams_continuing = self.new_beams(self.logprob_sums[best_continuing],
                                               self.tokens[best_continuing])
        best_beams_terminated = self.new_beams(self.logprob_sums[best_terminated],
                                               self.tokens[best_terminated])
        return best_beams_continuing, best_beams_terminated


    def print(self, max_print_chars=80):
        """
        Prints out a set of sequences with their corresponding logitsums.
        """
        if len(self.tokens) == 0:
            return

        output = []
        print('Best completions')

        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + text[-int(0.7 * max_print_chars):]
            print(f'logprob_sum: {logprob_sum:>8.3f}: {text}')


def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Path to the weights file', required=True)
    parser.add_argument('--sampling-method', type=str, choices=['greedy', 'beam'], required=True,
                        help='Method to sample next tokens from the transformer output')
    parser.add_argument('--prompt', type=str, help='Prompt to generate text with', required=True)
    return parser.parse_args()


def greedy_sample(model, input_tokens, max_tokens=40, eos_token_id=50256):
    """ Produce output text with the model by greedily sampling the maxium logits

    :param model: torch.nn.Module, transformer model
    :param input_tokens: torch.tensor, Input tokens to produce text with
    :param max_tokens: int, Maximum number of tokens to generate
    :param eos_token_id: int, id of the EOS token
    :return: torch.tensor, generated tokens
    """

    tokens = input_tokens.clone()
    model.eval()
    for i in range(max_tokens):
        # get logits of input_tokens
        logits = model(tokens)
        # use logits only for last token
        logits = logits[0, -1]
    
        # sample next token from logits
        next_token = logits.softmax(dim=-1).argmax().item()
        next_token = torch.tensor([next_token]).unsqueeze(0)
        # add next_token to input_tokens
        tokens = torch.cat([tokens, next_token], dim=-1)
        # if the next token is EOS, stop generating
        if next_token == eos_token_id:
            break
    return tokens[0]


def main():
    """ Main function """
    args = parse_args()
    
    # create tokenizer
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    # define model parameters
    n_layers = 6
    d_vocab = tokenizer.vocab_size
    context_length = 1024
    d_model = 256
    n_heads = 12
    d_head = 64

    # create model
    model = MiniGPT(n_layers, d_vocab, context_length, d_model, n_heads, d_head)

    # load weights
    weights_path = Path(args.weights)
    if not weights_path.exists() or not weights_path.is_file():
        print(f'Could not load weights file: {weights_path}! Please make sure it exists and' 
               'is a correct weights file')
        return

    try:
        model.load_state_dict(torch.load(weights_path), strict=True)
    except Exception as e:
        print(e)
        return
    
    # input prompt
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors='pt')

    # produce output text based on chosen method
    if args.sampling_method == 'greedy':
        generated_tokens = greedy_sample(model, prompt_tokens, max_tokens=40,
                                         eos_token_id=tokenizer.eos_token_id)
    elif args.sampling_method == 'beam':
        generated_tokens = beam_search(model, prompt_tokens, beam_size=3, max_length=8)

    # decode text
    generated_text = tokenizer.decode(generated_tokens)

    print(f'Input: {args.prompt}')
    print(f'Output: {generated_text}')

if __name__ == '__main__':
    main()
