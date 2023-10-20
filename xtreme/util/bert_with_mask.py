import math

import torch
from torch import nn as nn
from util.common import lang2id
from transformers.modeling_bert import BertSelfAttention, BertSelfOutput, BertAttention, BertIntermediate, BertOutput, BertEncoder, BertPooler, \
    BertEmbeddings, BertPreTrainingHeads, BertConfig, BertLayer, BertModel, BertPreTrainedModel, BertForPreTraining, BertForTokenClassification
from transformers.modeling_utils import prune_linear_layer
from util.other_utils import prev_curr


class BertConfigWithMask(BertConfig):
    masked_language: str = ""
    total_languages: int = len(lang2id)
    mask_val: int = 1
    mask_method: str = 'linear'
    lang2id: dict = lang2id
    prune_model: bool = False


def update_with_mask(tensor, mask, config: BertConfigWithMask):
    if config.prune_model: # mask is applied already to the pruned model
        return tensor

    lang_id = config.lang2id[config.masked_language]
    mask = mask[lang_id]

    if config.mask_method == 'constant':
        mask = torch.full_like(mask, config.mask_val)
    else:
        assert config.mask_method == 'linear'

    tensor *= mask

    return tensor


class BertSelfAttentionWithMask(BertSelfAttention):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        # This is to prevent accidental update of tensors during fine-tuning
        self.register_buffer('attention_mask', torch.zeros((config.total_languages, self.all_head_size)))
        self.register_buffer('value_attention_mask', torch.zeros((config.total_languages, self.all_head_size)))
        self.prune_id_list = None

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        mixed_query_layer = update_with_mask(mixed_query_layer, self.attention_mask, self.config)
        mixed_key_layer = update_with_mask(mixed_key_layer, self.attention_mask, self.config)
        mixed_value_layer = update_with_mask(mixed_value_layer, self.value_attention_mask, self.config)


        if self.config.prune_model:
            assert not self.output_attentions

            if self.prune_id_list is None:
                query_total_head_features = 0
                value_total_head_features = 0
                self.prune_id_list = [0]
                self.val_prune_id_list = [0]
                for head_id in range(self.num_attention_heads):
                    start_idx = head_id * self.attention_head_size
                    end_idx = (head_id + 1) * self.attention_head_size
                    ids = self.attention_mask[lang2id[self.config.masked_language]][start_idx:end_idx]
                    id_sum = int(ids.sum())
                    query_total_head_features += id_sum
                    self.prune_id_list.append(query_total_head_features)

                    start_idx = head_id * self.attention_head_size
                    end_idx = (head_id + 1) * self.attention_head_size
                    ids = self.value_attention_mask[lang2id[self.config.masked_language]][start_idx:end_idx]
                    id_sum = int(ids.sum())
                    value_total_head_features += id_sum
                    self.val_prune_id_list.append(value_total_head_features)

            context_layers = []
            for (q_start, q_end), (v_start, v_end) in zip(prev_curr(self.prune_id_list), prev_curr(self.val_prune_id_list)):
                query_layer = mixed_query_layer[:, :, q_start:q_end]
                key_layer = mixed_key_layer[:, :, q_start:q_end]
                value_layer = mixed_value_layer[:, :, v_start:v_end]

                # Take the dot product between "query" and "key" to get the raw attention scores.
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
                if attention_mask is not None:
                    # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                    attention_scores = attention_scores + attention_mask.squeeze(1)

                # Normalize the attention scores to probabilities.
                attention_probs = nn.Softmax(dim=-1)(attention_scores)

                # This is actually dropping out entire tokens to attend to, which might
                # seem a bit unusual, but is taken from the original Transformer paper.
                attention_probs = self.dropout(attention_probs)

                # Mask heads if we want to
                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                context_layer = torch.matmul(attention_probs, value_layer)
                context_layers.append(context_layer)

            context_layer = torch.cat(context_layers, dim=-1)
        else:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

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

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutputWithMask(BertSelfOutput):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.register_buffer('output_mask', torch.zeros((config.total_languages, config.hidden_size)))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = update_with_mask(hidden_states, self.output_mask, self.config)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttentionWithMask(BertAttention):
    def __init__(self, config: BertConfigWithMask):
        nn.Module.__init__(self)
        self.config = config
        self.self = BertSelfAttentionWithMask(config)
        self.output = BertSelfOutputWithMask(config)
        self.pruned_heads = set()
        self.pruned = False

    def prune(self):
        assert not self.pruned
        mask = self.self.attention_mask[lang2id[self.config.masked_language]].nonzero(as_tuple=True)[0]
        self.self.query = prune_linear_layer(self.self.query, mask)
        self.self.key = prune_linear_layer(self.self.key, mask)

        mask = self.self.value_attention_mask[lang2id[self.config.masked_language]].nonzero(as_tuple=True)[0]
        self.self.value = prune_linear_layer(self.self.value, mask)
        self.output.dense = prune_linear_layer(self.output.dense, mask, dim=1)

        self.pruned = True
        self.config.prune_model = True


class BertIntermediateWithMask(BertIntermediate):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        print(config.hidden_act)
        self.register_buffer('intermediate_mask', torch.zeros((config.total_languages, config.intermediate_size)))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = update_with_mask(hidden_states, self.intermediate_mask, self.config)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutputWithMask(BertOutput):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.register_buffer('layer_mask', torch.zeros((config.total_languages, config.hidden_size)))

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
        self.attention = BertAttentionWithMask(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttentionWithMask(config)
        self.intermediate = BertIntermediateWithMask(config)
        self.output = BertOutputWithMask(config)
        self.pruned = False

    def prune(self):
        assert not self.pruned
        self.attention.prune()
        mask = self.intermediate.intermediate_mask[lang2id[self.config.masked_language]].nonzero(as_tuple=True)[0]
        self.intermediate.dense = prune_linear_layer(self.intermediate.dense, mask)
        self.output.dense = prune_linear_layer(self.output.dense, mask, dim=1)
        self.pruned = True
        self.config.prune_model = True


class BertEncoderWithMask(BertEncoder):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        nn.Module.__init__(self)
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayerWithMask(config) for _ in range(config.num_hidden_layers)])

    def prune(self):
        for layer in self.layer:
            layer.prune()


class BertPoolerWithMask(BertPooler):
    def __init__(self, config: BertConfigWithMask):
        super().__init__(config)
        self.config = config
        self.register_buffer('pool_mask', torch.zeros((config.total_languages, config.hidden_size)))
        self.pruned = False

    def prune(self):
        assert not self.pruned
        mask = self.pool_mask[lang2id[self.config.masked_language]].nonzero(as_tuple=True)[0]
        self.dense = prune_linear_layer(self.dense, mask)
        self.pruned = True
        self.config.prune_model = True

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = update_with_mask(pooled_output, self.pool_mask, self.config)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModelWithMask(BertPreTrainedModel):
    config_class = BertConfigWithMask  # used in from_pretrained


class BertModelWithMask(BertModel, BertPreTrainedModelWithMask):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderWithMask(config)
        self.pooler = BertPoolerWithMask(config)

        self.init_weights()

    def prune(self):
        self.encoder.prune()
        self.pooler.prune()
        print("PRUNED!")


class BertForPreTrainingWithMask(BertForPreTraining, BertPreTrainedModelWithMask):
    config: BertConfigWithMask

    def __init__(self, config: BertConfigWithMask):
        BertPreTrainedModel.__init__(self, config)

        self.bert = BertModelWithMask(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()


class BertForTokenClassificationWithMask(BertForTokenClassification, BertPreTrainedModelWithMask):
    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.bert = BertModelWithMask(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
