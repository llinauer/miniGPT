"""
test.py 

Test the transformer model by letting it generate text
"""

import argparse
import torch
import transformers
from pathlib import Path
from model import MiniGPT

def parse_args():
    """ Parse the command-line args

    :return: argparse.Namespace obj, the parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Path to the weights file', required=True)
    parser.add_argument('--decoding-method', type=str, choices=['greedy', 'sample'],
                        help='Method to decode the transformer output', required=True)
    return parser.parse_args()

def greedy_search():
    """ Implement greedy search deconding method """
    raise NotImplementedError

def beam_search():
    """ Implement beam search deconding method """
    raise NotImplementedError

def sample(model, input_tokens, max_tokens=40, eos_token_id=50256):
    """ Produce output text with the model in an autoregressive way
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
        logits = model(input_tokens)
        # use logits only for last token
        logits = logits[0, -1]
        # sample from a categorical distribution with vocab_size possible outcomes and 
        # logits as weights
        next_token = torch.distributions.categorical.Categorical(logits=logits).sample().item()
        
        # add next_token to input_tokens
        tokens = torch.cat([input_tokens, next_token], dim=-1)

        # if the next token is EOS, stop generating
        if next_token == eos_token_id:
            break
    return tokens


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

    device = torch.device('cuda')
    
    # create model
    model = MiniGPT(n_layers, d_vocab, context_length, d_model, n_heads, d_head).to(device)

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

    # get decoding method
    if args.decoding_method == 'greedy':
        decoding_func = greedy_search
    elif args.decoding_method == 'beam':
        decoding_func = beam_search


if __name__ == '__main__':
    main()
