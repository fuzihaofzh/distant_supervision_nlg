from fairseq.models.transformer import *
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from fairseq.modules import LayerNorm, MultiheadAttention

class DecomposableTransformerModel(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DecomposableTransformerDecoder(args, tgt_dict, embed_tokens)

class DecomposableTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        decoder_layers = args.decoder_layers
        args.decoder_layers = 0
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            DecomposableTransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(decoder_layers)
        ])
        args.decoder_layers = decoder_layers
        self.user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None  for e in args.user_mode.split(',')}
        if 'sep_lm1' in self.user_mode:
            self.lm_embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
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
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **unused)
        x = self.output_layer(x, **unused)
        return x, extra

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
        for idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                layer_idx = idx,
                **unused
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
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                if 'use_lm_mode' in kwargs:
                    return F.linear(features, self.lm_embed_out)
                else:
                    return F.linear(features, self.embed_out)
        else:
            return features

class DecomposableTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.no_encoder_attn = no_encoder_attn
        mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None  for e in args.user_mode.split(',')}
        self.user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None  for e in args.user_mode.split(',')}
        if self.has_mode('2channel', 'sep_lm'):
            self.gate_fc1 = Linear(self.embed_dim * 2, self.embed_dim)
            self.gate_fc2 = Linear(self.embed_dim, 1)

        if self.is_mode('2channel'):
            export = getattr(args, 'char_inputs', False)
            self.lm_self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            self.lm_self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, self_attention=True
            )

        if self.has_mode('2channel', 'sep_lm1'):
            self.lm_layer_norm = LayerNorm(self.embed_dim)
            self.lm_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.lm_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
            self.lm_final_layer_norm = LayerNorm(self.embed_dim)

        if self.has_mode('sep_lm1'):
            self.bma_layer_norm = LayerNorm(self.embed_dim)
            self.bma_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.bma_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

    def is_mode(self, *args):
        return all([m in self.user_mode for m in args])

    def has_mode(self, *args):
        return any([m in self.user_mode for m in args])

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
        layer_idx = 0,
        **unused
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if self.is_mode('2channel'):
            if hasattr(x, 'y'):
                y = x.y
            else:
                y = x
        if self.is_mode('sep_lm1'):
            input_x = x

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

        if self.is_mode('2channel'):
            # maple lm
            lm_residual = y
            y = self.maybe_layer_norm(self.lm_self_attn_layer_norm, y, before=True)
            y, lm_attn = self.self_attn(
                query=y,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=None,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = lm_residual + y
            y = self.maybe_layer_norm(self.lm_self_attn_layer_norm, y, after=True)
        
        if self.is_mode('sep_lm1'):
            if 'use_lm_mode' in unused:
                x = self.feed_forward(x, self.lm_layer_norm, self.lm_fc1, self.lm_fc2)
                return x, attn
                

        if self.encoder_attn is not None and not self.no_encoder_attn:
            residual = x
            if self.is_mode('sep_lm'):
                x = unused['lm_out'][1]['inner_states'][layer_idx + 1]
                residual = x
            if self.is_mode('sep_lm1'):
                if 1 == 0:#layer_idx == 0:
                    residual = 0
                else:
                    x = self.feed_forward(x, self.bma_layer_norm, self.bma_fc1, self.bma_fc2)
                    residual = input_x
                    x = x + input_x
                

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
            x = F.dropout(x, p=self.dropout, training=self.training) 

            if self.has_mode('2channel'):
                residual = 0
            if self.is_mode('sep_lm'):
                if layer_idx == 0:
                    pass
                    #residual = 0
                #if layer_idx == 5:
                #    residual = unused['lm_out'][1]['inner_states'][layer_idx + 1]

            if self.has_mode('chg_add'):
                residual_bak = residual
                residual = 0

                
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

            # LM channel
            if self.is_mode('2channel'):
                lm_residual = y
                y = self.maybe_layer_norm(self.lm_layer_norm, y, before=True)
                y = self.activation_fn(self.lm_fc1(y))
                y = F.dropout(y, p=self.activation_dropout, training=self.training)
                y = self.lm_fc2(y)
                y = F.dropout(y, p=self.dropout, training=self.training)
                y = lm_residual + y
                y = self.maybe_layer_norm(self.lm_final_layer_norm, y, after=True)

                g = self.gate_fc2(self.gate_fc1(torch.cat([y, x], 2)).tanh()).sigmoid()
                x = g * y + (1 - g) * x

        residual = x
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

        if self.is_mode('2channel'):
            #x.g = g
            x.y = y
            #if layer_idx == 5:
            #g = self.gate_fc2(self.gate_fc1(torch.cat([y, x], 2)).tanh()).sigmoid()
            #x = g * y + (1 - g) * x
        if self.is_mode('sep_lm'):
            if layer_idx == 5:
                l = unused['lm_out'][1]['inner_states'][layer_idx + 1]
                g = self.gate_fc2(self.gate_fc1(torch.cat([l, x], 2)).tanh()).sigmoid()
                #x = g * l + (1 - g) * x

        if self.has_mode('chg_add'):
            if layer_idx == 5:
                y = self.feed_forward(residual_bak, self.bma_layer_norm, self.bma_fc1, self.bma_fc2)
                x = x + y

        return x, attn

    def feed_forward(self, x, layer_norm, fc1, fc2, use_dropout = True):
        residual = x
        x = self.maybe_layer_norm(layer_norm, x, before=True)
        x = self.activation_fn(fc1(x))
        if use_dropout:
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = fc2(x)
        if use_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(layer_norm, x, after=True)
        return x
