from fairseq.models.transformer import TransformerModel, register_model, TransformerEncoder, TransformerDecoder, register_model_architecture, Linear
from fairseq.models import BaseFairseqModel, FairseqEncoderDecoderModel, FairseqIncrementalDecoder
from undecorated import undecorated
from fairseq.models.transformer import base_architecture as base_architecture_transformer
from fairseq.models.dev_utils import *
from fairseq.models.proxy_transformer import *
from fairseq.models.sequence_generator_withgrad import SequenceGenerator as SequenceGeneratorGrad
from fairseq.sequence_generator import SequenceGenerator
from fairseq.models.gated_transformer import *
from fairseq.models.decomposable_transformer import *
from fairseq.models.endorsement_detector import *
from fairseq.models.endorsement_transformer import *
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import random
from itertools import *
from torch.nn.utils.rnn import pad_sequence

MODEL_NAME = "distant_transformer"

class UserTransformerDecoderLayer(SimpleTransformerDecoderLayer):
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
            need_weights=self.need_attn,#maple
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            attn1 = attn #maple
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
                need_weights= self.need_attn,# maple
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

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
        attn.prev_attn = attn1
        return x, attn

@register_model(MODEL_NAME)
class DistantTransformerModel(ProxyTransformerModel):

    def __init__(self, model):
        super().__init__(model)

    @classmethod
    def build_model(cls, args, task):
        mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None  for e in args.user_mode.split(',')}
        if 'gated' in mode:
            tmodel = GatedTransformerModel.build_model(args, task)
        elif any([m in mode for m in ['decomposable', 'sep_lm', 'sep_lm1']]):
            tmodel = DecomposableTransformerModel.build_model(args, task)
        elif any([m in mode for m in ['attn_endorse', 'dbg_log_endorsement']]):
            tmodel = SimpleTransformerModel.build_model(args, task, DecoderModelLayer = UserTransformerDecoderLayer)
        else:
            tmodel = SimpleTransformerModel.build_model(args, task)

        model = DistantTransformerModel(tmodel)
        model.args = args
        model.user_mode = mode
        model.sampler_grad = SequenceGeneratorGrad(model.model.decoder.dictionary, beam_size = 1, max_len_b = 60)
        model.sampler = SequenceGenerator(model.model.decoder.dictionary, beam_size = 1, max_len_b = 60)
        model.decoder = ProxyDecoder(tmodel, model.user_mode, args, task, model.sampler_grad, model.sampler)
        model.encoder = ProxyEncoder(tmodel, model.user_mode, args, task, model.sampler_grad, model.sampler)
        tmodel.encoder.user_mode = mode
        tmodel.decoder.user_mode = mode
        if any([m in mode for m in ['diff_lm', 'pretrain_lm', 'sep_lm', 'max_lm_margin', 'sep_lm2', 'sep_lm3']]):
            model.lm = TransformerDecoder(args, tmodel.decoder.dictionary, tmodel.decoder.embed_tokens, no_encoder_attn = True)
            model.decoder.lm = model.lm
        if 'sep_lm3' in mode:
            tmodel.decoder.gate_fc1 = Linear(len(tmodel.decoder.dictionary) * 2, len(tmodel.decoder.dictionary))
            tmodel.decoder.gate_fc2 = Linear(len(tmodel.decoder.dictionary), 1)
        if any([m in mode for m in ['endorsement', 'rl_edm', 'beam_endorse']]):
            model.edm = EndorsementDetectorModel.build_model(args, task)
            model.decoder.edm = model.encoder.edm = model.edm
            model.encoder.edm.decoder.user_mode = model.encoder.edm.encoder.user_mode = mode
            if any([m in mode for m in ['self_align']]):
                model.self_edm = EndorsementDetectorModel.build_model(args, task)
                model.decoder.self_edm = model.encoder.self_edm = model.self_edm
                model.encoder.self_edm.decoder.user_mode = model.encoder.self_edm.encoder.user_mode = mode

        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        if hasattr(self, 'epoch_id'):
            self.encoder.epoch_id = self.epoch_id
            self.decoder.epoch_id = self.epoch_id
        if hasattr(self, 'batch_id'):
            self.encoder.batch_id = self.batch_id
            self.decoder.batch_id = self.batch_id
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, prev_output_tokens = prev_output_tokens, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


class ProxyEncoder(ProxyModule): 
    def __init__(self, model, uer_mode, args, task, sampler_grad, sampler):
        super(ProxyEncoder, self).__init__(model, uer_mode, args, task)
        self.sampler_grad = sampler_grad
        self.sampler = sampler
        if self.has_mode('simple_append'):
            nsize = 2048
            head = 1
            self.head = head
            self.nsize = nsize
            self.M = torch.nn.Parameter(torch.randn(head, args.encoder_embed_dim, nsize))
            self.Mk = torch.nn.Parameter(torch.randn(head, args.encoder_embed_dim, args.encoder_embed_dim)) 
            self.Mv = torch.nn.Parameter(torch.randn(head, args.encoder_embed_dim, args.encoder_embed_dim))
            self.Mq = torch.nn.Parameter(torch.randn(head, args.encoder_embed_dim, args.encoder_embed_dim))


    def forward(self, src_tokens, src_lengths, prev_output_tokens = None, **kwargs):
        if self.has_mode('base'):
            encoder_out = self.model.encoder(src_tokens, src_lengths)
        elif self.has_mode('rl_edm'):
            encoder_out = self.model.encoder(src_tokens, src_lengths)
            if self.model.training:
                edm_encoder_out = self.edm.encoder(src_tokens, src_lengths, **kwargs)
                encoder_out['edm_encoder_out'] = edm_encoder_out
                encoder_out['edm_encoder_out']['src_tokens'] = src_tokens
                encoder_out['edm_encoder_out']['src_lengths'] = src_lengths
        elif self.has_mode('simple_append'):
            encoder_out = self.model.encoder(src_tokens, src_lengths, **kwargs)
            if self.training and random.random() < 0.2:
                emean = encoder_out['encoder_out'].sum(0) / math.sqrt(encoder_out['encoder_out'].shape[0])
                Q = emean.unsqueeze(0).expand(self.Mq.shape[0], *emean.shape).bmm(self.Mq) / math.sqrt(self.Mq.shape[1])
                K = self.Mk.bmm(self.M) / math.sqrt(self.M.shape[1])
                V = self.Mv.bmm(self.M) / math.sqrt(self.M.shape[1])
                E = F.softmax(Q.bmm(K) / math.sqrt(K.shape[1]) , -1).bmm(V.transpose(1, 2))
                E = F.layer_norm(E, (E.shape[-1],))
                torch.cat([encoder_out['encoder_out'][:-1, :, :], E, encoder_out['encoder_out'][-1:, :, :]])
                encoder_out['encoder_out'] = torch.cat([encoder_out['encoder_out'][:-1, :, :], E, encoder_out['encoder_out'][-1:, :, :]])
                if encoder_out['encoder_padding_mask'] is not None:
                    epm = encoder_out['encoder_padding_mask']
                    pd = epm.new(epm.shape[0], self.head)
                    pd[:] = False
                    encoder_out['encoder_padding_mask'] = torch.cat([epm, pd], 1)#'''
        elif self.has_mode('rl_word'):
            encoder_out = self.model.encoder(src_tokens, src_lengths, **kwargs)
            tokens, scores = self._sample_sequence(src_tokens, src_lengths, ['tokens', 'positional_scores'])
            if prev_output_tokens is not None:
                tokens_correct = self.a_in_b(tokens, prev_output_tokens).float()
                mask = tokens.ne(self.pad).float()
                encoder_out['rl_loss'] = - mask * scores * (tokens_correct.float() - 0.9)#tokens_correct.sum() / mask.sum())
            torch.cuda.empty_cache()
            print((mask * (tokens_correct == 0).float()).sum())
        elif self.has_mode('max_lm_margin'):
            if self.model.training:
                shash = (src_tokens.ne(self.pad).long() * src_tokens).sum(1) % 10
                idx1 = idx2 = (shash > -1000).nonzero().squeeze(-1)
                encoder_out = self.model.encoder(src_tokens[idx1], src_lengths[idx1], **kwargs)
                encoder_out_heldout = self.model.encoder(src_tokens[idx2], src_lengths[idx2], **kwargs)
                encoder_out['idx1'] = idx1
                encoder_out['idx2'] = idx2
                encoder_out['encoder_out_heldout'] = encoder_out_heldout
            else:
                encoder_out = self.model.encoder(src_tokens, src_lengths, **kwargs)            
        elif self.has_mode('pretrain_lm'):
            encoder_out = {'encoder_out': None, 'encoder_padding_mask': None}
        elif self.has_mode('endorsement'):
            if self.has_mode('pretrain'):
                edm_encoder_out = self.edm.encoder(src_tokens, src_lengths, **kwargs)
                encoder_out = {}
                if self.has_mode('self_align'):
                    encoder_out['self_align_encoder_out'] = self.self_edm.encoder(prev_output_tokens, None, **kwargs)
                    encoder_out['self_align_encoder_out']['src_tokens'] = prev_output_tokens

            else:
                encoder_out = self.model.encoder(src_tokens, src_lengths)
                edm_encoder_out = {}
                if self.model.training:
                    edm_encoder_out = self.edm.encoder(src_tokens, src_lengths, **kwargs)
                    if self.has_mode('self_align'):
                        encoder_out['self_align_encoder_out'] = self.self_edm.encoder(prev_output_tokens, None, **kwargs)
                        encoder_out['self_align_encoder_out']['src_tokens'] = prev_output_tokens
            encoder_out['edm_encoder_out'] = edm_encoder_out
            encoder_out['edm_encoder_out']['src_tokens'] = src_tokens
            encoder_out['edm_encoder_out']['src_lengths'] = src_lengths
        
            
        else:
            #print("No such user_mode!!! exit!")
            #exit(-1)
            encoder_out = self.model.encoder(src_tokens, src_lengths)
        encoder_out['src_tokens'] = src_tokens
        encoder_out['src_lengths'] = src_lengths
        return encoder_out


    def reorder_encoder_out(self, *args):
        return self.model.encoder.reorder_encoder_out(*args)

class ProxyDecoder(ProxyModule):
    def __init__(self, model, uer_mode, args, task, sampler_grad, sampler):
        super(ProxyDecoder, self).__init__(model, uer_mode, args, task)
        self.sampler_grad = sampler_grad
        self.sampler = sampler
        if self.has_mode('sep_lm2'):
            self.gate_fc1 = Linear(len(self.model.decoder.dictionary) * 2, len(self.model.decoder.dictionary))
            self.gate_fc2 = Linear(len(self.model.decoder.dictionary), len(self.model.decoder.dictionary))
    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        if self.is_mode('pretrain_lm'):
            decoder_out = self.lm(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out[1]['user_mode'] = self.user_mode
            return decoder_out
        elif self.has_mode('endorsement'):
            if self.has_mode('pretrain'):
                decoder_out = self.endorsement_decoder(kwargs['target'], encoder_out, **kwargs)
                if self.has_mode('self_align'):
                    decoder_out[1]['self_align_decoder_out'] = self.endorsement_decoder(kwargs['target'], encoder_out, edm_model = self.self_edm, encoder_out_name = 'self_align_encoder_out', remove_exact = True, **kwargs)
            elif self.has_mode('mask_prev'):
                extra = {}
                if self.model.training:
                    self.edm.eval()
                    prev_dem_decoder_out = self.endorsement_decoder(prev_output_tokens, encoder_out, **kwargs)
                    extra = {'weight': prev_dem_decoder_out[1]['weight']}

                decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **extra)
                if self.model.training:
                    edm_decoder_out = self.endorsement_decoder(kwargs['target'], encoder_out, **kwargs)
                    decoder_out[1]['edm_decoder_out'] = edm_decoder_out
            elif self.has_mode('drop_words'):
                if self.model.training:
                    self.edm.eval()
                    edm_decoder_out = self.endorsement_decoder(kwargs['target'], encoder_out, **kwargs)
                    prev_output_tokens, drop_words_target, new_length = self.drop_words(prev_output_tokens, edm_decoder_out)
                decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
                if self.model.training:
                    decoder_out[1]['drop_words_prev_output_tokens'] = prev_output_tokens
                    decoder_out[1]['drop_words_target'] = drop_words_target
            else:# vanila endorsement
                decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
                if self.model.training:
                    self.edm.eval()
                    edm_decoder_out = self.endorsement_decoder(kwargs['target'], encoder_out, **kwargs)
                    decoder_out[1]['edm_decoder_out'] = edm_decoder_out
                    if self.has_mode('add_exact'):
                        decoder_out[1]['word_exactness'] = self.calc_exact_endorse(prev_output_tokens.get_device())
                    if self.has_mode('self_align'):
                        self.self_edm.eval()
                        decoder_out[1]['self_edm_decoder_out'] = self.endorsement_decoder(kwargs['target'], encoder_out, edm_model=self.self_edm, encoder_out_name = 'self_align_encoder_out',remove_exact = True, **kwargs)

        elif self.is_mode('sep_lm'):
            lm_out = self.lm(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            kwargs['lm_out'] = lm_out
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out[1]['lm_out'] = lm_out
        elif self.is_mode('sep_lm1'):
            lm_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, use_lm_mode = True)
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out[1]['lm_out'] = lm_out
        elif self.is_mode('diff_lm'):
            lm_out = self.lm(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        elif self.has_mode('max_lm_margin'):
            if self.model.training:
                idx1, idx2, encoder_out_heldout = encoder_out['idx1'], encoder_out['idx2'], encoder_out['encoder_out_heldout']
                lm_out = self.lm(prev_output_tokens[idx1], encoder_out=None, **kwargs)
                decoder_out = self.model.decoder(prev_output_tokens[idx1], encoder_out=encoder_out, **kwargs)

                lm_out_margin = self.lm(prev_output_tokens[idx2], encoder_out=None, **kwargs)
                decoder_out_margin = self.model.decoder(prev_output_tokens[idx2], encoder_out=encoder_out_heldout, **kwargs)
                decoder_out[1]['lm_out'] = lm_out
                decoder_out[1]['lm_out_margin'] = lm_out_margin
                decoder_out[1]['decoder_out_margin'] = decoder_out_margin

            else:
                decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        elif self.is_mode('add_lm'):
            for layer in self.model.decoder.layers:
                layer.no_encoder_attn = True
            lm_out = self.model.decoder(prev_output_tokens, **kwargs)
            for layer in self.model.decoder.layers:
                layer.no_encoder_attn = False
            decoder_out[1]['lm_out'] = lm_out
        elif self.is_mode('sep_lm2'):
            lm_out = self.lm(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out[1]['lm_out'] = lm_out
            logden = decoder_out[0].logsumexp(-1, keepdim = True)
            p_dec = decoder_out[0].softmax(-1)
            p_lm = lm_out[0].softmax(-1)
            logit = (p_dec - p_lm.detach()).clamp(min = 5e-8).log() + logden
            g = self.gate_fc2(self.gate_fc1(torch.cat([lm_out[0], logit], 2)).tanh()).sigmoid()
            merged = lm_out[0] * g + logit * (1 - g)
            
            decoder_out = (merged, decoder_out[1])
        elif self.is_mode('sep_lm3'):
            torch.cuda.empty_cache()
            lm_out = self.lm(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            #g = self.model.decoder.gate_fc2(self.model.decoder.gate_fc1(torch.cat([lm_out[0], decoder_out[0]], 2)).tanh()).sigmoid()
            if not self.has_mode('pretrain'):
                g = self.model.decoder.gate_fc2(torch.stack([lm_out[0], decoder_out[0]], 0)).softmax(0)
                logit = g[0] * lm_out[0] + g[1] * decoder_out[0]
                decoder_out = (logit, decoder_out[1])
                decoder_out[1]['gate'] = g
            decoder_out[1]['lm_out'] = lm_out
        elif self.is_mode('rl_edm'):
            if self.model.training:
                src_tokens, src_lengths = encoder_out['src_tokens'], encoder_out['src_lengths']
                tokens, scores = _sample_sequence(self, src_tokens, src_lengths, ['tokens', 'positional_scores'])
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            if self.model.training:
                self.edm.eval()
                edm_decoder_out = self.endorsement_decoder(tokens, encoder_out, **kwargs)
                decoder_out[1]['word_lprob'] = scores
                decoder_out[1]['word_tokens'] = tokens
                decoder_out[1]['word_scores'] = edm_decoder_out[1]['max_match'].detach()
        else:
            decoder_out = self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)

        decoder_out[1]['encoder_out'] = encoder_out
        decoder_out[1]['user_mode'] = self.user_mode
        return decoder_out

    def endorsement_decoder_v1(self, prev_output_tokens, encoder_out, **kwargs):
        decoder_out = self.edm.decoder(prev_output_tokens, encoder_out=encoder_out['edm_encoder_out'], **kwargs)
        decoder_out[1]['p_mask'] = prev_output_tokens.ne(self.pad)
        if self.edm.training:
            if not hasattr(self, 'record_epoch'):
                self.record_epoch = self.epoch_id
            if self.epoch_id == self.record_epoch:
                if not hasattr(self, 'tgts'):
                    self.tgts = []
                self.tgts += [i for i in prev_output_tokens.view(-1).cpu().numpy().tolist() if i > 2]
            #make neg
            prev_output_tokens_list = prev_output_tokens.cpu().numpy().tolist()
            neg_prev_output_tokens_list = []
            for row in prev_output_tokens_list:
                swords = set(row)
                neg_row = random.choices(self.tgts, k = len(row))
                for i in range(len(neg_row))[::-1]:
                    if row[i] == self.pad: neg_row[i] = self.pad
                    elif row[i] == self.eos: neg_row[i] = self.eos
                    else:
                        while neg_row[i] in swords:
                            neg_row[i] = random.choice(self.tgts)
                neg_prev_output_tokens_list.append(neg_row)
            neg_prev_output_tokens = prev_output_tokens.new(neg_prev_output_tokens_list)

            neg_out = self.edm.decoder(neg_prev_output_tokens, encoder_out=encoder_out['edm_encoder_out'], **kwargs)
        
        
            decoder_out[1]['neg_out'] = neg_out
            decoder_out[1]['n_mask'] = neg_prev_output_tokens.ne(self.pad)
            decoder_out[1]['n_in_p_mask'] = self.a_in_b(neg_prev_output_tokens, prev_output_tokens)
        
        return decoder_out

    def endorsement_decoder_v0(self, prev_output_tokens, encoder_out, **kwargs):
        decoder_out = self.edm.decoder(prev_output_tokens, encoder_out=encoder_out['edm_encoder_out'], **kwargs)
        decoder_out[1]['p_mask'] = prev_output_tokens.ne(self.pad)
        if self.edm.training:
            if not hasattr(self, 'record_epoch'):
                self.record_epoch = self.epoch_id
            if self.epoch_id == self.record_epoch:
                if not hasattr(self, 'tgts'):
                    self.tgts = []
                self.tgts.append(prev_output_tokens)
            neg_prev_output_tokens = random.sample(self.tgts, 1)[0]
            while neg_prev_output_tokens.shape[0] < prev_output_tokens.shape[0]:
                nsp = random.sample(self.tgts, 1)[0]
                maxlen = max(neg_prev_output_tokens.shape[1], nsp.shape[1])
                nsp = F.pad(nsp, (0, maxlen - nsp.shape[1]), value = self.pad)
                neg_prev_output_tokens = F.pad(neg_prev_output_tokens, (0, maxlen - neg_prev_output_tokens.shape[1]), value = self.pad)
                neg_prev_output_tokens = torch.cat([nsp, neg_prev_output_tokens])
            ids = torch.randperm(neg_prev_output_tokens.shape[0])
            neg_prev_output_tokens = neg_prev_output_tokens[ids[:prev_output_tokens.shape[0]]]

            neg_out = self.edm.decoder(neg_prev_output_tokens, encoder_out=encoder_out['edm_encoder_out'], **kwargs)
        
        
            decoder_out[1]['neg_out'] = neg_out
            decoder_out[1]['n_mask'] = neg_prev_output_tokens.ne(self.pad)
            decoder_out[1]['n_in_p_mask'] = self.a_in_b(neg_prev_output_tokens, prev_output_tokens)
        
        return decoder_out

    def endorsement_decoder(self, prev_output_tokens, encoder_out, edm_model = None, encoder_out_name = 'edm_encoder_out', remove_exact = False, **kwargs):
        if not edm_model:
            edm_model = self.edm
        decoder_out = edm_model.decoder(prev_output_tokens, encoder_out=encoder_out[encoder_out_name], remove_exact = remove_exact, **kwargs)
        decoder_out[1]['p_mask'] = prev_output_tokens.ne(self.pad)
        if edm_model.training:
            if not hasattr(self, 'record_epoch'):
                self.record_epoch = self.epoch_id
            if self.epoch_id == self.record_epoch:
                if not hasattr(self, 'tgts'):
                    self.tgts = []
                self.tgts.append(prev_output_tokens)
            neg_prev_output_tokens = random.sample(self.tgts, 1)[0]
            while neg_prev_output_tokens.shape[0] < prev_output_tokens.shape[0]:
                nsp = random.sample(self.tgts, 1)[0]
                maxlen = max(neg_prev_output_tokens.shape[1], nsp.shape[1])
                nsp = F.pad(nsp, (0, maxlen - nsp.shape[1]), value = self.pad)
                neg_prev_output_tokens = F.pad(neg_prev_output_tokens, (0, maxlen - neg_prev_output_tokens.shape[1]), value = self.pad)
                neg_prev_output_tokens = torch.cat([nsp, neg_prev_output_tokens])
            ids = torch.randperm(neg_prev_output_tokens.shape[0])
            neg_prev_output_tokens = neg_prev_output_tokens[ids[:prev_output_tokens.shape[0]]]

            neg_out = edm_model.decoder(neg_prev_output_tokens, encoder_out=encoder_out[encoder_out_name], remove_exact = remove_exact, **kwargs)
        
        
            decoder_out[1]['neg_out'] = neg_out
            decoder_out[1]['n_mask'] = neg_prev_output_tokens.ne(self.pad)
            decoder_out[1]['n_in_p_mask'] = self.a_in_b(neg_prev_output_tokens, prev_output_tokens)

            # ploss & nloss
            pe = decoder_out[0].float()
            ne = neg_out[0].float()
            p_mask = decoder_out[1]['p_mask']
            n_mask = decoder_out[1]['n_mask']
            n_in_p_mask = decoder_out[1]['n_in_p_mask'].float()
            decoder_out[1]['ploss'] = (-F.logsigmoid(pe).masked_select(p_mask)).sum()
            decoder_out[1]['nloss'] = (-F.logsigmoid(-ne*(1 - n_in_p_mask)).masked_select(n_mask)).sum()
        
        return decoder_out

    def get_words_endorse(self, src, scores, threshold):
        results = []
        for i in range(src.shape[0]):
            results.append(' '.join([self.model.decoder.dictionary.symbols[w] + ('*' if scores[i][j] < threshold else '') for j, w in enumerate(src[i])]))
        return results

    def get_align(self, src_tokens, prev_output_tokens, decoder_out1):
        m = decoder_out1['m']
        s_mask_inf = decoder_out1['s_mask_inf']
        aligns = []
        srcs = self.get_src_words(src_tokens)
        tgts = self.get_tgt_words(prev_output_tokens)
        for i in range(src_tokens.shape[0]):
            sent = []
            for j, t in enumerate(prev_output_tokens[i]):
                k = m[i, :, j].argmax()
                sent.append([self.model.encoder.dictionary[src_tokens[i, k]], self.model.encoder.dictionary[t], k.item(), j, m[i, k, j].item()])
            aligns.append([sent, srcs[i], tgts[i]])
        return aligns

    def calc_exact_endorse(self, cuda_id):
        if not hasattr(self, 'word_exactness'):
            epoch_itr = self.task.get_batch_iterator(
                dataset=self.task.dataset(self.args.train_subset),
                max_tokens=self.args.max_tokens,
                max_sentences=self.args.max_sentences,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    self.model.max_positions(),
                ),
                ignore_invalid_inputs=True,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_shards=self.args.distributed_world_size,
                shard_id=self.args.distributed_rank,
                num_workers=self.args.num_workers,
                epoch=0,
            )
            itr = epoch_itr.next_epoch_itr(
                fix_batches_to_gpus=self.args.fix_batches_to_gpus,
                shuffle=(epoch_itr.epoch >= self.args.curriculum),
            )
            exact_cnt = torch.tensor([0.] * len(self.model.decoder.dictionary))
            total_cnt = torch.tensor([0.] * len(self.model.decoder.dictionary))
            for samples in itr:
                src = samples['net_input']['src_tokens']
                tgt = samples['target']
                tgt_in_src = self.a_in_b(tgt, src).bool()
                for i in range(len(tgt)):
                    total_cnt[tgt[i][tgt[i].ne(self.pad)]] += 1
                    exact_cnt[tgt[i][tgt_in_src[i]]] += 1
            word_exactness = exact_cnt / total_cnt.clamp_min(1)
            self.word_exactness = word_exactness.cuda(cuda_id)
        return self.word_exactness

    def drop_words(self, prev_output_tokens, edm_decoder_out):
        keep = (edm_decoder_out[1]['weight'] > torch.rand_like(edm_decoder_out[1]['weight']))
        npad = prev_output_tokens.ne(self.pad) & prev_output_tokens.ne(self.eos)
        keep = keep & npad
        keep_list = []
        for i in range(keep.shape[0]):
            tokens = prev_output_tokens[i].masked_select(keep[i])
            keep_list.append(tokens)
        tokens1 = pad_sequence(keep_list, batch_first = True, padding_value = self.pad)
        new_length = tokens1.ne(self.pad).sum(1)
        new_prev = torch.cat([tokens1.new_ones(tokens1.shape[0]).unsqueeze(1) * self.eos, tokens1], dim = 1)
        new_target = torch.cat([tokens1, tokens1.new_ones(tokens1.shape[0]).unsqueeze(1) * self.pad], dim = 1)
        new_target[torch.arange(new_target.shape[0]), new_length] = self.eos
        new_length += 1
        return new_prev, new_target, new_length
        



@register_model_architecture(MODEL_NAME, MODEL_NAME)
def base_architecture_proxy_transformer(args):
    undecorated(base_architecture_transformer)(args)

def _sample_sequence(self, src_tokens, src_lengths, keys = None):
    sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}
    beam_output = self.sampler_grad.generate([self.model], sample)
    if keys is None:
        keys = beam_output[0][0].keys()
    output = [[b[0][k] for k in keys] for b in beam_output]
    result = list(zip(*output))
    result = [pad_sequence(r, batch_first = True, padding_value = self.pad) for r in result]
    return result