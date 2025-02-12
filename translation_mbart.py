import torch
import torch.nn as nn
import wget
import json
import os
import sentencepiece as spm
import re

TRANSLATION_FOLDER = "./TranslationModel"
TRANSLATION_MODEL_WEIGHTS_FILE = "pytorch_model.bin"
TRANSLATION_MODEL_CONFIG_FILE = "config.json"
TRANSLATION_MODEL_VOCAB_FILE = "sentencepiece.bpe.model"
TRANSLATION_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/pytorch_model.bin"
TRANSLATION_MODEL_CONFIG_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/config.json"
TRANSLATION_MODEL_VOCAB_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model"
TRANSLATION_MODEL_FILES_URLS = [
    (TRANSLATION_MODEL_WEIGHTS_URL, TRANSLATION_MODEL_WEIGHTS_FILE),
    (TRANSLATION_MODEL_CONFIG_URL, TRANSLATION_MODEL_CONFIG_FILE),
    (TRANSLATION_MODEL_VOCAB_URL, TRANSLATION_MODEL_VOCAB_FILE),
]
TRANSLATION_SPM_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model"
TRANSLATION_SPM = "sentencepiece.bpe.model"

def ensure_translation_files_exist():
    os.makedirs(TRANSLATION_FOLDER, exist_ok=True)
    for url, filename in TRANSLATION_MODEL_FILES_URLS:
        filepath = os.path.join(TRANSLATION_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)
    filepath_spm = os.path.join(TRANSLATION_FOLDER, TRANSLATION_SPM)
    if not os.path.exists(filepath_spm):
        wget.download(TRANSLATION_SPM_URL, out=filepath_spm)

class MBartConfig:
    def __init__(self, vocab_size, hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, intermediate_size=4096, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, layer_norm_eps=1e-05, initializer_range=0.02, pad_token_id=1, bos_token_id=0, eos_token_id=2, n_positions=1024, n_ctx=1024, decoder_layers=12, decoder_attention_heads=16, decoder_ffn_dim=4096, encoder_layers=12, encoder_attention_heads=16, encoder_ffn_dim=4096, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class MBartForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = MBartModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.final_logits_bias = nn.Parameter(torch.zeros((1, config.vocab_size)))

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        return lm_logits

class MBartModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MBartEncoder(config)
        self.decoder = MBartDecoder(config)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask)
        return decoder_outputs

class MBartEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_positions = MBartSinusoidalPositionalEmbedding(config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = self.embed_positions(input_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layernorm_embedding(embeddings)
        encoder_states = embeddings
        all_encoder_layers = []
        for layer_module in self.layers:
            encoder_states = layer_module(encoder_states, encoder_padding_mask=attention_mask)
            all_encoder_layers.append(encoder_states)
        return (encoder_states, all_encoder_layers)

class MBartDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_positions = MBartSinusoidalPositionalEmbedding(config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.hidden_size)

    def forward(self, decoder_input_ids, encoder_outputs, decoder_attention_mask=None):
        inputs_embeds = self.embed_tokens(decoder_input_ids)
        position_embeddings = self.embed_positions(decoder_input_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layernorm_embedding(embeddings)
        decoder_states = embeddings
        all_decoder_layers = []
        all_cross_attention_layers = []
        for layer_module in self.layers:
            decoder_states, cross_attn_weights = layer_module(decoder_states, encoder_outputs[0], decoder_padding_mask=decoder_attention_mask, encoder_padding_mask=encoder_outputs[0])
            all_decoder_layers.append(decoder_states)
            all_cross_attention_layers.append(cross_attn_weights)
        return (decoder_states, all_decoder_layers, all_cross_attention_layers)

class MBartSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(self.padding_idx + 1, seq_len + self.padding_idx + 1, dtype=torch.long, device=input_ids.device)
        return self.get_embedding(positions)

    def get_embedding(self, positions):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=positions.device) * -emb)
        emb = torch.outer(positions.float(), emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

class MBartEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MBartAttention(config, embed_dim=config.hidden_size, num_heads=config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, encoder_padding_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attention_mask=encoder_padding_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.relu(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states

class MBartDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MBartAttention(config, embed_dim=config.hidden_size, num_heads=config.decoder_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.encoder_attn = MBartAttention(config, embed_dim=config.hidden_size, num_heads=config.decoder_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, encoder_hidden_states, decoder_padding_mask=None, encoder_padding_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attention_mask=decoder_padding_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights = self.encoder_attn(hidden_states, encoder_hidden_states, encoder_hidden_states, attention_mask=encoder_padding_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.relu(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states, cross_attn_weights

class MBartAttention(nn.Module):
    def __init__(self, config, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, query, key, value, attention_mask=None):
        bsz, tgt_len, _ = query.size()
        bsz, src_len, _ = key.size()
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        query = self._shape(query, tgt_len, bsz)
        key = self._shape(key, src_len, bsz)
        value = self._shape(value, src_len, bsz)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class MBartTokenizer:
    def __init__(self, sentencepiece_processor):
        self.sp = sentencepiece_processor
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.model_max_length = 1024

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=None, src_lang="en_XX", tgt_lang="es_XX", **kwargs):
        max_length = max_length if max_length is not None else self.model_max_length
        self.sp.SetEncodeExtraOptions("bos:<s>,eos:</s>")
        input_ids = self.sp.EncodeAsIds(f"{src_lang} {text}")
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        if padding:
            input_ids += [self.pad_token_id] * (max_length - len(input_ids))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.ones(len(input_ids)).unsqueeze(0)}
        return input_ids

    def batch_decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, target_lang="es_XX"):
        decoded_texts = []
        for ids in token_ids:
            text = self.sp.DecodeIds(list(ids))
            if skip_special_tokens:
                text = re.sub(r'(<s>|</s>|<pad>)', '', text).strip()
            if clean_up_tokenization_spaces:
                text = text.replace(' ', ' ').strip()
            decoded_texts.append(text.replace(f"{target_lang} ", ""))
        return decoded_texts