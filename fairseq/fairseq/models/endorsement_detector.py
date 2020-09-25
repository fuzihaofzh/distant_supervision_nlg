from fairseq.models.transformer import *
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from fairseq.modules import LayerNorm, MultiheadAttention

class EndorsementDetectorModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return EndorsementDetectorEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return EndorsementDetectorDecoder(args, tgt_dict, embed_tokens)

class EndorsementDetectorEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        encoder_layers = args.encoder_layers
        args.encoder_layers = 0
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            EndorsementDetectorEncoderLayer(args)
            for i in range(1)
        ])
        args.encoder_layers = encoder_layers

        self.eds_fc1 = Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False)
        self.eds_fc2 = Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False)
        self.eds_layer_norm = LayerNorm(args.encoder_embed_dim)

        self.self_attn = MultiheadAttention(
            args.encoder_embed_dim, args.encoder_attention_heads,
            dropout=0, self_attention=True
        )

    def forward(self, src_tokens, src_lengths, **unused):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if 'nnwa' in self.user_mode:
            x = self.embed_tokens(src_tokens)
            x = self.eds_fc2(torch.tanh(self.eds_fc1(x)))
            #x = self.eds_layer_norm(x)            
        elif 'simple_attn' in self.user_mode:
            x = self.embed_scale * self.embed_tokens(src_tokens)
            x = x.transpose(0, 1)
            residual = x
            #x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
            #x = residual + x
            x = self.eds_layer_norm(x)
            x = self.eds_fc2(F.relu(self.eds_fc1(x)))
            x = self.eds_layer_norm(x)
        else:
            x = extract_sent_features(self, src_tokens)

        return {
            'encoder_out': x,  # B x T x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class EndorsementDetectorDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        decoder_layers = args.decoder_layers
        args.decoder_layers = 0
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            EndorsementDetectorDecoderLayer(args, no_encoder_attn)
            for _ in range(1)
        ])
        args.decoder_layers = decoder_layers
        self.padding_idx = embed_tokens.padding_idx
        self.proj_prob = Linear(args.decoder_embed_dim, 1, bias=True)
        self.eds_fc1 = Linear(args.decoder_embed_dim, args.decoder_embed_dim, bias=False)
        self.eds_fc2 = Linear(args.decoder_embed_dim, args.decoder_embed_dim, bias=False)
        self.eds_layer_norm = LayerNorm(args.decoder_embed_dim)

        self.self_attn = MultiheadAttention(
            args.decoder_embed_dim, args.decoder_attention_heads,
            dropout=0, self_attention=True
        )


    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, remove_exact = False, **unused):
        s_mask_01 = encoder_out['src_tokens'].ne(self.padding_idx)
        s_mask_inf = torch.zeros_like(s_mask_01).half()
        s_mask_inf[s_mask_01 == 0] = float('-inf')
        t_mask_01 = prev_output_tokens.ne(self.padding_idx)
        t_mask_inf = torch.zeros_like(t_mask_01).half()
        t_mask_inf[t_mask_01 == 0] = float('-inf')
        extra = {}
        if 'nnwa' in self.user_mode:
            t = self.embed_tokens(prev_output_tokens)
            t = self.eds_fc2(torch.tanh(self.eds_fc1(t)))
            #t = self.eds_layer_norm(t)
            t = t.transpose(2, 1)
            s = encoder_out['encoder_out']#.transpose(1, 0)
            m = s.matmul(t)
            r = 1.0
            #aggr = (m + s_mask_inf.unsqueeze(-1)).logsumexp(1)
            #aggr = (m * s_mask_01.unsqueeze(-1).half()).sum(1)
            #aggr = (m + s_mask_inf.unsqueeze(-1)).max(1)[0]
            aggr = ((m + s_mask_inf.unsqueeze(-1)).softmax(1) * m).sum(1)
        elif 'simple_attn' in self.user_mode:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            x = x.transpose(0, 1)
            residual = x
            #x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=prev_output_tokens.eq(self.padding_idx))
            #x = residual + x
            x = self.eds_layer_norm(x)
            x = self.eds_fc2(F.relu(self.eds_fc1(x)))
            x = self.eds_layer_norm(x)

            t = x.permute(1, 2, 0)
            s = encoder_out['encoder_out'].transpose(1, 0)
            m = s.matmul(t)
            aggr = (m + s_mask_inf.unsqueeze(-1)).logsumexp(1)
            extra = {"s" : s, "t": t, "m" : m}
        else:
            x = extract_sent_features(self, prev_output_tokens)

            t = x.permute(1, 2, 0)
            s = encoder_out['encoder_out'].transpose(1, 0)
            m = s.matmul(t)
            aggr = (m + s_mask_inf.unsqueeze(-1)).logsumexp(1)


            # direct word align
            wid_s = encoder_out['src_tokens'].unsqueeze(-1)
            wid_t = prev_output_tokens.unsqueeze(1)
            weq = (wid_s == wid_t) * s_mask_01.unsqueeze(-1) * t_mask_01.unsqueeze(1)
            logp = (m + s_mask_inf.unsqueeze(-1)).log_softmax(1)
            wloss = -logp.masked_select(weq)

            # penalty on most frequent src
            # avoid align to one word too much
            freq_loss = (m.sum(2)).max(1)[0].sigmoid()

            max_match = (m + s_mask_inf.unsqueeze(-1)).max(1)[0]
            # calc weight
            if remove_exact:
                m[wid_s == wid_t] = float("-inf")
                aggr = (m + s_mask_inf.unsqueeze(-1)).logsumexp(1)
            a, b = (float(self.user_mode['a']), float(self.user_mode['b'])) if 'a' in self.user_mode else (1.0, 0.0)
            weight = (a * max_match - b).sigmoid().float()

            extra = {"s" : s, "t": t, "m" : m, "wloss" : wloss, "freq_loss" : freq_loss, "s_mask_inf": s_mask_inf, "t_mask_01": t_mask_01, "weight" : weight, "max_match": max_match}

        return aggr, extra



    def forward_attn(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x.squeeze(-1), extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                #self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        logit = self.proj_prob(features)
        return logit
        '''if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features'''

def extract_sent_features(self, tokens):
    x = self.embed_scale * self.embed_tokens(tokens)
    x = x.transpose(0, 1)
    residual = x
    #x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=prev_output_tokens.eq(self.padding_idx))
    #x = residual + x
    x = self.eds_layer_norm(x)
    x = self.eds_fc2(F.relu(self.eds_fc1(x)))
    x = self.eds_layer_norm(x)
    return x

class EndorsementDetectorEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

class EndorsementDetectorDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            return x, attn
            #x = F.dropout(x, p=self.dropout, training=self.training)
            #x = residual + x
            #x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        '''residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn'''
        