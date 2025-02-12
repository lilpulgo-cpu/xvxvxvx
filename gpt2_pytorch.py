import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wget
import json
from tqdm import tqdm

GPT2_FOLDER = "./GPT2"
MODEL_FILE = "gpt2-pytorch_model.bin"
ENCODER_FILE = "encoder.json"
VOCAB_FILE = "vocab.bpe"
MODEL_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin"
ENCODER_URL = "https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/refs/heads/master/GPT2/GPT2/encoder.json"
VOCAB_URL = "https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/refs/heads/master/GPT2/GPT2/vocab.bpe"
MAX_LENGTH = 1024
END_OF_TEXT_TOKEN = "<|endoftext|>"

def ensure_gpt2_files_exist():
    if not os.path.exists(os.path.join(GPT2_FOLDER, MODEL_FILE)):
        wget.download(MODEL_URL, out=os.path.join(GPT2_FOLDER, MODEL_FILE))
    if not os.path.exists(os.path.join(GPT2_FOLDER, ENCODER_FILE)):
        wget.download(ENCODER_URL, out=os.path.join(GPT2_FOLDER, ENCODER_FILE))
    if not os.path.exists(os.path.join(GPT2_FOLDER, VOCAB_FILE)):
        wget.download(VOCAB_URL, out=os.path.join(GPT2_FOLDER, VOCAB_FILE))

class GPT2Config:
    def __init__(self, vocab_size_or_config_json_file=50257, n_positions=MAX_LENGTH, n_ctx=MAX_LENGTH, n_embd=768, n_layer=12, n_head=12, layer_norm_epsilon=1e-5, initializer_range=0.02):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        lm_logits, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        return lm_logits, presents

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e-10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))