# miniGPT
MiniGPT - A small language model

This repository is an attempt on my part, to better understand transformer models, 
especially of the GPT flavour.
It is heavily inspired by (not to say stolen from) Neel Nandas' "Transformers from Scratch" tutorial, which can be found 
on his website https://www.neelnanda.io/.
A remake of this tutorial can be found here: https://arena-ch1-transformers.streamlit.app/[1.1]_Transformer_from_Scratch

I take no credit for the ideas presented here.

## Pre-requisites

To run the code in this repo, you will need a python3.9+ installation and the packages
from the requirements.txt

The easiest way to get started is to create a python virtualenv or a conda environment with python3.9
and then run

    pip install -r requirements.txt

## Basics

### Transformers

The Transformer is a neural network architecture introduced by Vaswani et al in the paper
"Attention is all you need" (https://arxiv.org/abs/1706.03762).
Since then, it became hugely popular and was used in many many many applications
for language modeling, most notably the GPT models. 
The GPT (Generative Pre-Trained) transformer model was introduced by OpenAI in 2018 (https://openai.com/research/language-unsupervised).
As of now (2023), ChatGPT, an application that uses a GPT-like model to generate language is so incredibly 
wide-known, even representatives of the austrian government know what it is.
And representatives of the austrian government are not known to be particularly tech-savvy.

There are some differences between specific transformer architectures, but I will focus on the GPT-style
transformer here.
On a very high level, the transformer architecture looks like this:

![Transformer Architecture](transformer_architecture.jpg)

First, an embedding layer translates text input to floating-point numbers. These embeddings are 
the transformer-internal representation of the text. The output of the embedding layer is called residual
stream. It reaches from the input of the transformer all the way down to the output.
Each transformer layer gets its input from the residual stream and adds its output back to it.
It is the backbone of the transformer so to speak. 
After the embedding, the input is passed through a series of attention heads. Each head performs some
operation and then adds its output back to the residual stream.
The number of heads is a hyperparameter. The heads together constitute what is known as an Attention Layer.
After the attention layer, the residual stream passes through an MLP (multi-layer perceptron).
This is an ordinary densely connected neural network. The output of the MLP is again added to the 
residual stream.
These two layers, the Attention Layer, and the MLP Layer form a Transformer Block.
There may be many of these Transformer Blocks inside a specific model.
Finally, the residual stream passes through an Unembedding layer, where it is transformed back to text.

This is of course a very simplified representation. There are numerous details to consider, e.g. does
the transformer not operate on text directly, you usually want to add some normalization in between 
the layers, you need to specify several more hyperparameters like the size of the attention heads and
the MLP hidden dimension. Also, the transformer does not simply output text, but probabilities about
possible next tokens in a sequence.

The architecture is specified in gpt_mini/model.py You can go check out the details there.


