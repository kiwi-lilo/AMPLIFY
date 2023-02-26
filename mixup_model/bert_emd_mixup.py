from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertPreTrainedModel
import torch

class bert_emd_mixup(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _forward_init(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        return input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, \
            encoder_hidden_states, encoder_attention_mask, extended_attention_mask, encoder_extended_attention_mask


    def forward(self, x1, att1, x2, att2, lam):
        x1, attention_mask1, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, \
            encoder_attention_mask, extended_attention_mask1, encoder_extended_attention_mask = self._forward_init(
                input_ids=x1, attention_mask=att1)
        embedding_output1 = self.embeddings(
            input_ids=x1, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        x2, attention_mask2, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, \
            encoder_attention_mask, extended_attention_mask2, encoder_extended_attention_mask = self._forward_init(
                input_ids=x2, attention_mask=att2)

        embedding_output2 = self.embeddings(
            input_ids=x2, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        embedding_output = lam * embedding_output1 + (1.0 - lam) * embedding_output2

        # need to take max of both to ensure we don't miss attending to any value
        extended_attention_mask = torch.max(extended_attention_mask1, extended_attention_mask2)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    class TextBERT(nn.Module):
        def __init__(self, pretrained_model, num_class, fine_tune, dropout):
            super(TextBERT, self).__init__()
            self.output_dim = num_class
            self.bert = MyBertModel.from_pretrained(pretrained_model)
            # Freeze bert layers
            if not fine_tune:
                for p in self.bert.parameters():
                    p.requires_grad = False

            bert_dim = MODELS[pretrained_model][2]
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(bert_dim, num_class)

        def forward(self, x, attn_masks):
            outputs = self.bert(x, attention_mask=attn_masks)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

        def forward_mix_embed(self, x1, att1, x2, att2, lam):
            outputs = self.bert.forward_mix_embed(x1, att1, x2, att2, lam)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

        def forward_mix_sent(self, x1, att1, x2, att2, lam):
            logits1 = self.forward(x1, att1)
            logits2 = self.forward(x2, att2)
            y = lam * logits1 + (1.0-lam) * logits2
            return y

        def forward_mix_encoder(self, x1, att1, x2, att2, lam):
            outputs1 = self.bert(x1, att1)
            outputs2 = self.bert(x2, att2)
            pooled_output1 = self.dropout(outputs1[1])
            pooled_output2 = self.dropout(outputs2[1])
            pooled_output = lam * pooled_output1 + (1.0-lam) * pooled_output2
            y = self.classifier(pooled_output)
            return y
