import torch
import torch.nn as nn
import wget
import json
import os

SENTIMENT_FOLDER = "./SentimentModel"
SENTIMENT_MODEL_WEIGHTS = "pytorch_model.bin"
SENTIMENT_VOCAB = "sentiment_vocab.json"
SENTIMENT_CONFIG = "config.json"
SENTIMENT_MODEL_WEIGHTS_URL = "https://huggingface.co/cardiffnlp/distilroberta-base-sentiment/resolve/main/pytorch_model.bin"
SENTIMENT_VOCAB_URL = "https://huggingface.co/cardiffnlp/distilroberta-base-sentiment/resolve/main/vocab.json"
SENTIMENT_CONFIG_URL = "https://huggingface.co/cardiffnlp/distilroberta-base-sentiment/resolve/main/config.json"
SENTIMENT_FILES_URLS = [
    (SENTIMENT_MODEL_WEIGHTS_URL, SENTIMENT_MODEL_WEIGHTS),
    (SENTIMENT_VOCAB_URL, SENTIMENT_VOCAB),
    (SENTIMENT_CONFIG_URL, SENTIMENT_CONFIG),
]

def ensure_sentiment_files_exist():
    os.makedirs(SENTIMENT_FOLDER, exist_ok=True)
    for url, filename in SENTIMENT_FILES_URLS:
        filepath = os.path.join(SENTIMENT_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, num_labels)

    def forward(self, sequence_output):
        x = sequence_output[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask)
        return (encoder_outputs[0], )

class RobertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        input_embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)
            all_encoder_layers.append(hidden_states)
        return (hidden_states, all_encoder_layers)

class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self_attn(hidden_states, attention_mask=attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.all_head_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states