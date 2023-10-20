import math

import torch
from torch import nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput, BertAttention, BertIntermediate, BertOutput, BertEncoder, BertPooler, \
    BertEmbeddings, BertPreTrainingHeads, BertConfig, BertLayer, BertModel, BertPreTrainedModel, BertForPreTraining, BertForTokenClassification

from common import lang2id

import torch.nn.functional as F


class BertConfigWithMask(BertConfig):
    masked_language: str = ""
    total_languages: int = len(lang2id)
    hard_threshold: float = None
    mask_method: str = 'linear'
    mask_val: int = 1

def update_with_mask(tensor, mask, config: BertConfigWithMask):
    mask = mask[lang2id[config.masked_language]]

    if config.mask_method == 'constant':
        mask = torch.full_like(mask, config.mask_val)
    else:
        assert config.mask_method == 'linear'

    tensor *= mask

    return tensor


class BertSelfAttentionWithMask(BertSelfAttention):
    def __init__(self, config: BertConfigWithMask, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.config = config
        # This is to prevent accidental update of tensors during fine-tuning
        self.attention_mask = torch.nn.Parameter(torch.zeros((config.total_languages, self.all_head_size)))
        self.value_attention_mask = torch.nn.Parameter(torch.zeros((config.total_languages, self.all_head_size)))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        mixed_query_layer = update_with_mask(mixed_query_layer, self.attention_mask, self.config)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            raise NotImplementedError
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            raise NotImplementedError
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            raise NotImplementedError
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            mixed_key_layer = update_with_mask(self.key(hidden_states), self.attention_mask, self.config)
            mixed_value_layer = update_with_mask(self.value(hidden_states), self.value_attention_mask, self.config)

            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)


        query_layer = self.transpose_for_scores(mixed_query_layer)


        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutputWithMask(BertSelfOutput):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.output_mask = torch.nn.Parameter(torch.zeros((config.total_languages, config.hidden_size)))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = update_with_mask(hidden_states, self.output_mask, self.config)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttentionWithMask(BertAttention):
    def __init__(self, config: BertConfigWithMask, position_embedding_type=None):
        nn.Module.__init__(self)
        self.config = config
        self.self = BertSelfAttentionWithMask(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutputWithMask(config)
        self.pruned_heads = set()


class BertIntermediateWithMask(BertIntermediate):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.intermediate_mask = torch.nn.Parameter(torch.zeros((config.total_languages, config.intermediate_size)))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = update_with_mask(hidden_states, self.intermediate_mask, self.config)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutputWithMask(BertOutput):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.layer_mask = torch.nn.Parameter(torch.zeros((config.total_languages, config.hidden_size)))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = update_with_mask(hidden_states, self.layer_mask, self.config)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class BertLayerWithMask(BertLayer):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        nn.Module.__init__(self)
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionWithMask(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttentionWithMask(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediateWithMask(config)
        self.output = BertOutputWithMask(config)


class BertEncoderWithMask(BertEncoder):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        self.config = config
        nn.Module.__init__(self)
        self.layer = nn.ModuleList([BertLayerWithMask(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

class BertPoolerWithMask(BertPooler):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.pool_mask = torch.nn.Parameter(torch.zeros((config.total_languages, config.hidden_size)))

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = update_with_mask(pooled_output, self.pool_mask, self.config)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModelWithMask(BertPreTrainedModel):
    config_class = BertConfigWithMask # used in from_pretrained


class BertModelWithMask(BertModel, BertPreTrainedModelWithMask):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask, add_pooling_layer=True):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderWithMask(config)

        self.pooler = BertPoolerWithMask(config) if add_pooling_layer else None

        self.init_weights()


class BertForPreTrainingWithMask(BertForPreTraining, BertPreTrainedModelWithMask):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        BertPreTrainedModel.__init__(self, config)

        self.bert = BertModelWithMask(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()


class BertForTokenClassificationWithMask(BertForTokenClassification, BertPreTrainedModelWithMask):
    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.bert = BertModelWithMask(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
