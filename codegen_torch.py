import torch
import torch.nn as nn
import torch.nn.functional as F
import wget
import json
import os
import sentencepiece as spm
import re

CODEGEN_FOLDER = "./CodeGenModel"
CODEGEN_MODEL_NAME = "codegen-350M-multi"
CODEGEN_MODEL_WEIGHTS = "pytorch_model.bin"
CODEGEN_CONFIG = "config.json"
CODEGEN_VOCAB = "vocab.json"
CODEGEN_MERGES = "merges.txt"
CODEGEN_MODEL_WEIGHTS_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/pytorch_model.bin"
CODEGEN_CONFIG_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/config.json"
CODEGEN_VOCAB_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/vocab.json"
CODEGEN_MERGES_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/merges.txt"
CODEGEN_FILES_URLS = [
    (CODEGEN_MODEL_WEIGHTS_URL, CODEGEN_MODEL_WEIGHTS),
    (CODEGEN_CONFIG_URL, CODEGEN_CONFIG),
    (CODEGEN_VOCAB_URL, CODEGEN_VOCAB),
    (CODEGEN_MERGES_URL, CODEGEN_MERGES),
]
CODEGEN_SPM_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/spm.model"
CODEGEN_SPM = "spm.model"

def ensure_codegen_files_exist():
    os.makedirs(CODEGEN_FOLDER, exist_ok=True)
    for url, filename in CODEGEN_FILES_URLS:
        filepath = os.path.join(CODEGEN_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)
    filepath_spm = os.path.join(CODEGEN_FOLDER, CODEGEN_SPM)
    if not os.path.exists(filepath_spm):
        wget.download(CODEGEN_SPM_URL, out=filepath_spm)

class CodeGenConfig:
    def __init__(self, vocab_size, n_positions=2048, n_ctx=2048, n_embd=1024, n_layer=24, n_head=16, n_inner=None, activation_function="gelu_new", resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, scale_attn_weights=True, use_cache=True, bos_token_id=50256, eos_token_id=50256, **kwargs):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class CodeGenForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = CodeGenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(transformer_outputs)
        return logits

class CodeGenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CodeGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states.view(*output_shape)

class CodeGenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CodeGenAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = CodeGenMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feedforward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feedforward_hidden_states
        return hidden_states

class CodeGenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class CodeGenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.embed_dim = config.n_embd
        self.split_size = self.embed_dim
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale_attn_weights = config.scale_attn_weights
        self.use_cache = config.use_cache
        self.register_buffer("bias", torch.tril(torch.ones((config.n_ctx, config.n_ctx), dtype=torch.uint8)).view((1, 1, config.n_ctx, config.n_ctx)))

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(value.size(-1))

        mask = self.bias[:, :, :attn_weights.size(-2), :attn_weights.size(-1)]
        attn_weights = torch.where(mask.bool(), attn_weights, torch.tensor(-1e4, device=attn_weights.device))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(*new_shape)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_value=None, use_cache=False):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.n_head, self.embed_dim // self.n_head)
        key = self._split_heads(key, self.n_head, self.embed_dim // self.n_head)
        value = self._split_heads(value, self.n_head, self.embed_dim // self.n_head)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present_key_value = (key, value) if use_cache else None
        attn_output = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.n_head, self.embed_dim // self.n_head)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present_key_value)
        return outputs[0]