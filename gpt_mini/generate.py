"""
test.py

Test the transformer model by letting it generate text
"""

import argparse
from pathlib import Path
import torch
import transformers
import einops
from transformer_lens import EasyTransformer
from tqdm import tqdm
from model import MiniGPT


class Beams:
    """Class to store beams during beam search."""

    def __init__(self, model, tokenizer, logprob_sums, tokens):
        """ Constructor
        :param model: nn.Module, transformer model
        :param tokenizer: transformers.tokenizer, tokenizer to use
        :param logprob_sums: torch.tensor(n_beams), current logprob sums of all beams
        :param tokens: torch.tensor(n_beams, seq_length), current best sampling tokens
        :return: None
        """

        self.model = model
        self.tokenizer = tokenizer
        self. logprob_sums = logprob_sums
        self.tokens = tokens

    def new_beams(self, logprob_sums, tokens):
        """Creates a new Beams object with updated logprob_sums and tokens
        :param logprob_sums: torch.tensor(n_beams), current logprob sums of all beams
        :param tokens: torch.tensor(n_beams, seq_length), current best sampling tokens
        :return: Beams object
        """
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx):
        """Allows you to take a slice of the beams object along the batch dimension
        :param idx: int/list/slice, index the beams
        :return: Beams object
        """
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self):
        """Returns self as a list of logprob sums and completions
        :return: list, list of logprob sums and sampled tokens
        """
        return [(logprob_sum.item(), self.tokenizer.decode(tokens)) for
                (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)]

    def generate(self, toks_per_beam, no_repeat_ngram_size=None):
        """ Starting from the current set of beams, returns a new
            set, containing the best `toks_per_beam` continuations for each of the original beams.

        :param toks_per_beam: int, how many tokens should be sampled for each beam
        :param no_repeat_ngram_size: int, Optional (default=`None`)
                                     the maximum length of repeating n-grams to be allowed
        """

        # get logits and logprobs
        logits = self.model(self.tokens)[:, -1, :]
        logprobs = logits.log_softmax(dim=-1)

        # get the top k logprobs and their indices; if no_repeat_ngram_size is given,
        # use get_topk_non_repeating instead of topk
        if no_repeat_ngram_size is None:
            topk_logprobs, topk_token_idcs = logprobs.topk(k=toks_per_beam)
        else:
            topk_logprobs, topk_token_idcs = self.get_topk_non_repeating(
                logprobs, no_repeat_ngram_size, toks_per_beam)

        # unpack and flatten the logprob sums
        logprob_sums_unpacked = einops.repeat(self.logprob_sums, "batch -> batch k",
                                              k=toks_per_beam).flatten()
        # flatten the topk logprobs
        topk_logprobs_flattened = einops.rearrange(topk_logprobs, "batch k -> (batch k)")
        # sum the two to generate the new logprob sums
        new_logprob_sums = sum([logprob_sums_unpacked, topk_logprobs_flattened])

        # generate new tokens
        new_tokens = torch.concat([einops.repeat(self.tokens, "batch seq -> (batch k) seq",
                                                 k=toks_per_beam),
                                   einops.rearrange(topk_token_idcs, "batch k -> (batch k) 1")
                                  ], dim=-1)

        return self.new_beams(new_logprob_sums, new_tokens)


    def filter(self, num_beams):
        """ Filter the top `num_beams`
        :param num_beams: int, how many beams to keep
        :return: best_beams: Beams, best `num_beams` which are not terminated.
                 early_terminations: Beams, best `num_beams` which are also terminated.
        """

        # get the indices of top `num_beams` beams
        top_beam_indices = self.logprob_sums.topk(k=num_beams, dim=0).indices.tolist()
        # get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = torch.nonzero(new_tokens == self.tokenizer.eos_token_id)

        # get the indices of the `num_beams` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        # return the beam objects from these indices
        best_beams_continuing = self.new_beams(self.logprob_sums[best_continuing],
                                               self.tokens[best_continuing])
        best_beams_terminated = self.new_beams(self.logprob_sums[best_terminated],
                                               self.tokens[best_terminated])
        return best_beams_continuing, best_beams_terminated


    def get_topk_non_repeating(self, logprobs, no_repeat_ngram_size, k):
        """ Generate new tokens but exclude those sequences, which have `no_repeat_ngram_size`
            repeating n-grams in it (e.g. the the the = repeating 3-gram)
        :param logprobs: torch.tensor(n_beams, vocab_size), logprobs of all beams
        :param no_repeat_ngram_size: int, the maximum allowed length of repeating n-grams
        :param k: int, number of top logits to return for each beam
        :return: output of logprobs.topk, without repeating n-grams
        """

        batch, seq_len = self.tokens.shape
        neg_inf = torch.tensor(-1.0e4)

        # if the number of non-repeatinng n-grams is None or too high, do nothing
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size-1):

            # check for ngram repetitions
            # first, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size-1):]

            # next, find all the tokens we're not allowed to generate (by going iterating through
            # past ngrams and seeing if those ngram prefixes match the last one)
            for i in range(seq_len - (no_repeat_ngram_size-1)):
                ngrams = self.tokens[:, i:i+no_repeat_ngram_size] # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1) # (batch,)
                ngram_end_tokens = ngrams[:, [-1]] # (batch, 1)

                # fill logprobs with neg_inf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = torch.where(
                    ngrams_are_repeated,
                    neg_inf,
                    logprobs[range(batch), ngram_end_tokens],
                )

        return logprobs.topk(k=k, dim=-1)


    def print(self, max_print_chars=80):
        """ Prints out a set of sequences with their corresponding logprob sums
        :param max_print_chars: int, maximum number of characters to print for each sequence
        :return: None
        """

        # nothing to print
        if len(self.tokens) == 0:
            return

        print('Best completions')

        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + \
                    text[-int(0.7 * max_print_chars):]
            print(f'logprob_sum: {logprob_sum:>8.3f}: {text}')


def beam_search(model, tokenizer, prompt, num_return_sequences, num_beams, max_new_tokens,
                no_repeat_ngram_size=None, verbose=False):
    """ Implements a beam search, by repeatedly performing the `generate` and `filter` steps
        (starting from the initial prompt) until either of the two stopping criteria are met:
        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.
    :param model: nn.Module, transformer model
    :param tokenizer: transformers.tokenizer, tokenizer to use
    :param prompt: str, input prompt
    :param num_return_sequences: int, number of returned sampled sequences
    :param num_beams: int, number of beams to keep during search
    :param max_new_tokens: int, maximum number of tokens to generate per beam
    :param no_repeat_ngram_size: int, Optional (default=None)
                                 maximum allowed length of repeating n-grams
    :param verbose: bool, if True, produce verbose output
    :return: list, list of top `num_return_sequences` and according sampled sequences
    """

    assert num_return_sequences <= num_beams
    model.eval()

    tokens = tokenizer.encode(prompt, return_tensors="pt")

    # lList for final beams to return (and early terminations)
    final_logprobs_and_completions = []
    # keep track of all best beams after each step
    best_beams = Beams(model, tokenizer, torch.tensor([0.0]), tokens)

    # loop until we have max_new_tokens
    for _ in tqdm(range(max_new_tokens)):

        # generation step
        best_beams = best_beams.generate(toks_per_beam=num_beams,
                                         no_repeat_ngram_size=no_repeat_ngram_size)

        # filtering step
        best_beams, best_beams_terminated = best_beams.filter(num_beams=num_beams)
        final_logprobs_and_completions.extend(best_beams_terminated.logprobs_and_completions)

        # print output
        if verbose:
            best_beams.print()

        # check stopping condition
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    final_logprobs_and_completions = final_logprobs_and_completions[:num_return_sequences]

    return final_logprobs_and_completions


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
    for _ in range(max_tokens):
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


def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Path to the weights file')
    parser.add_argument('--use-gpt2-small', action='store_true',
                        help='If given, use the weights from a fully trained GPT2 instance')
    parser.add_argument('--sampling-method', type=str, choices=['greedy', 'beam'], required=True,
                        help='Method to sample next tokens from the transformer output')
    parser.add_argument('--prompt', type=str, help='Prompt to generate text with', required=True)
    return parser.parse_args()


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

    # load weights
    # if --use-gpt2-small is given
    if args.use_gpt2_small:
        reference_gpt2 = EasyTransformer.from_pretrained(
            'gpt2-small', fold_ln=False, center_unembed=False, center_writing_weights=False)
        model = MiniGPT(12, reference_gpt2.tokenizer.vocab_size, 1024, 768, 12, 64)
        model = load_gpt2_weights(model, reference_gpt2)
    else:
        # create model
        model = MiniGPT(n_layers, d_vocab, context_length, d_model, n_heads, d_head)
        weights_path = Path(args.weights)
        if not weights_path.exists() or not weights_path.is_file():
            print(f'Could not load weights file: {weights_path}! Please make sure it exists and'
                   'is a correct weights file')
            return
    
        try:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(weights_path), strict=True)
            else:
                model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        except Exception as e:
            print(e)
            return

    # input prompt
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors='pt')

    # produce output text based on chosen method
    if args.sampling_method == 'greedy':
        generated_tokens = greedy_sample(model, prompt_tokens, max_tokens=40,
                                         eos_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_tokens)

    elif args.sampling_method == 'beam':
        top_beams = beam_search(model, tokenizer, args.prompt, num_return_sequences=3,
                                num_beams=40, max_new_tokens=60, no_repeat_ngram_size=2,
                                verbose=False)
        generated_text = top_beams[0][1]

    print(f'Input: {args.prompt}')
    print(f'Output: {generated_text}')


if __name__ == '__main__':
    main()
