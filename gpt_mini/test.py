"""
test.py 

Test the transformer model by letting it generate text
"""

import argparse
import torch
import transformers
from pathlib import Path
import einops
from model import MiniGPT

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



def beam_search(model, input_tokens, tokens_per_beam, max_tokens=40, eos_token_id=50256,
                device='cuda'):
    """ Beam search sampling method
        
    :param model: torch.nn.Module, transformer model
    :param input_tokens: torch.tensor, Input tokens to produce text with
    :param sampling_function: function, function to sample next token from logits
    :param max_tokens: int, Maximum number of tokens to generate
    :param eos_token_id: int, id of the EOS token
    :return: torch.tensor, generated tokens
    """

    
    tokens = input_tokens.clone().to(device)
    logprob_sums = torch.tensor([.0]).to(device)
    model.eval()
    #for i in range(max_tokens):
        
    
    # get logits from model 
    logits = model(tokens)
    # get logprobs from logits, but only for last tokens in the batch
    logprobs = logits[:, -1, :].log_softmax(dim=-1)

    # get the top tokens_per_beam tokens for each beam
    topk_logprobs, topk_tokens = logprobs.topk(k=tokens_per_beam)

    # get new logprob sums
    new_logprob_sums = sum([einops.repeat(logprob_sums, 'batch -> batch k', k=tokens_per_beam),
                            einops.rearrange(topk_logprobs, 'batch k -> (batch k)')])
    # get new tokens for each beam
    new_tokens = torch.concat([einops.repeat(tokens, 'batch seq -> (batch k) seq',
                                             k=tokens_per_beam),
                               einops.rearrange(topk_tokens, 'batch k -> (batch k) 1')],
                               dim=-1)





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
        next_token = logits.argmax().item()
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
    
    # input prompt
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors='pt')

    # produce output text based on chosen method
    if args.sampling_method == 'greedy':
        generated_tokens = greedy_sample(model, prompt_tokens, max_tokens=40,
                                         eos_token_id=tokenizer.eos_token_id)
    elif args.sampling_method == 'beam':
        generated_tokens = beam_search(model, prompt_tokens, tokens_per_beam=3, max_tokens=40,
                                       eos_token_id=tokenizer.eos_token_id)

    # decode text
    generated_text = tokenizer.decode(generated_tokens)

    print(f'Input: {args.prompt}')
    print(f'Output: {generated_text}')

if __name__ == '__main__':
    main()
