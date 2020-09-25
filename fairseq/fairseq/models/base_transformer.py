from fairseq.models.transformer import *
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer



class BaseTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, UserTransformerDecoderLayer, no_encoder_attn=False, user_mode = None):
        decoder_layers = args.decoder_layers
        args.decoder_layers = 0
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            UserTransformerDecoderLayer(args, no_encoder_attn, user_mode = user_mode)
            for _ in range(decoder_layers)
        ])
        args.decoder_layers = decoder_layers
        self.user_mode = user_mode

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **unused)
        x = self.output_layer(x)
        return x, extra

    def is_mode(self, *args):
        return all([m in self.user_mode for m in args])

    def has_mode(self, *args):
        return any([m in self.user_mode for m in args])

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        x = self.extract_features_embedding(prev_output_tokens, incremental_state)
        x, inner_states, attn = self.extract_features_decoder_layers(x, incremental_state, encoder_out)
        x = self.extract_features_output(x)
        return x, {'attn': attn, 'inner_states': inner_states}

    def extract_features_embedding(self, prev_output_tokens, incremental_state):
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
        return x

    def extract_features_decoder_layers(self, x, incremental_state, encoder_out):
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
        return x, inner_states, attn

    def extract_features_output(self, x):
        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x


class BaseTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, user_mode = {}):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.user_mode = user_mode

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
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        
        x, attn = self.forward_masked_self_attention(x, prev_self_attn_state, incremental_state, self_attn_padding_mask, self_attn_mask)
        x, attn = self.forward_query_encoder_out(x, prev_attn_state, incremental_state, encoder_padding_mask, encoder_out, attn)
        result = self.forward_ffn(x, attn)
        return result    

        

    def forward_masked_self_attention(self, x, prev_self_attn_state, incremental_state, self_attn_padding_mask, self_attn_mask):
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
        return x, attn

    def forward_query_encoder_out(self, x, prev_attn_state, incremental_state, encoder_padding_mask, encoder_out, attn):
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
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        return x, attn

    def forward_ffn(self, x, attn):
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
        return x, attn
    
#def register_component(Decoder = BaseTransformerDecoder, DecoderLayer = BaseTransformerDecoderLayer):
class BaseTransformerModel(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        self.user_mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None  for e in args.user_mode.split(',')}
        return BaseTransformerDecoder(args, tgt_dict, embed_tokens, BaseTransformerDecoderLayer, user_mode = self.user_mode)
#return BaseTransformerModel