from fairseq.models.base_transformer import *



class EndorsementTransformerDecoder(BaseTransformerDecoder):
    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        # embed positions
        x = self.extract_features_embedding(prev_output_tokens, incremental_state)
        if self.has_mode('mask_prev') and 'weight' in unused:
            weight = unused['weight']
            x.weight = weight.log()
        x, inner_states, attn = self.extract_features_decoder_layers(x, incremental_state, encoder_out)
        x = self.extract_features_output(x)
        return x, {'attn': attn, 'inner_states': inner_states}

    def buffered_future_mask(self, tensor):
        mask = super().buffered_future_mask(tensor)
        if self.has_mode('mask_prev') and hasattr(tensor, 'weight'):
            weight = tensor.weight.detach().half().unsqueeze(1)
            weight[weight == float('-inf')] = 0
            mask = mask.unsqueeze(0) + weight
            #del tensor.weight # only for first layer
        return mask

class EndorsementTransformerDecoderLayer(BaseTransformerDecoderLayer):
    pass

class EndorsementTransformerModel(BaseTransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return EndorsementTransformerDecoder(args, tgt_dict, embed_tokens, EndorsementTransformerDecoderLayer)