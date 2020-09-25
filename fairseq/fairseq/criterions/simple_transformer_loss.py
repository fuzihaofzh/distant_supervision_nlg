import math
import torch
from fairseq import utils
import torch.nn.functional as F 
from . import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss


@register_criterion('distant_transformer_loss')
class SimpleTransformerLoss(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)

    def is_mode(self, *args):
        return all([m in self.user_mode for m in args])

    def has_mode(self, *args):
        return any([m in self.user_mode for m in args])

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample['net_input']['target'] = sample['target']
        net_output = model(**(sample['net_input']))
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        self.user_mode = net_output[1]["user_mode"]
        if self.is_mode('max_lm_margin'):
            if self.training:
                lm_out_margin, decoder_out_margin = net_output[1]['lm_out_margin'], net_output[1]['decoder_out_margin']
                lp = self.get_log_probs(lm_out_margin, model).detach()
                lq =self.get_log_probs(decoder_out_margin, model)
                idx1 = net_output[1]['encoder_out']['idx1']
                idx2 = net_output[1]['encoder_out']['idx2']
                nkl = (lp.exp() * (lq - lp)).sum() / len(idx2) * len(idx1)
                new_sample = self.sub_sample_by_id(sample, idx1, model)
                loss, nll_loss = super().compute_loss(model, net_output, new_sample)
                loss_lm, nll_loss_lm = super().compute_loss(model, net_output[1]['lm_out'], new_sample)
                loss = loss + float(self.user_mode['rkl']) * nkl + loss_lm
            else:
                loss, nll_loss = super().compute_loss(model, net_output, sample)
            return loss, nll_loss
        elif self.is_mode('simple_mask'):
            msk = model.decoder.a_in_b(sample['target'], sample['net_input']['src_tokens'])
            mask = (msk | ((torch.rand_like(msk.float()) > float(self.user_mode['simple_mask'])) & (sample['target'].ne(model.decoder.pad))).long()).float()
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1, 1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=None, reduce=False,
            )
            loss, nll_loss = (loss * mask.view(-1)).sum(), (nll_loss * mask.view(-1)).sum()
            return loss, nll_loss
        elif self.is_mode('endorsement', 'lse'):
            pe = net_output[0].float()
            ne = net_output[1]['neg_out'][0].float()
            p_mask = net_output[1]['p_mask'].float()
            n_mask = net_output[1]['n_mask'].float()
            loss = ((1 + torch.exp(-pe)).log() * p_mask).sum() + ((1 + torch.exp(ne)).log() * n_mask).sum()
            #pe[pe == float('-inf')] = float('inf')
            #loss = ((1 + torch.exp(-pe)).log() * p_mask).sum() + ((1 + torch.exp(ne)).log() * n_mask).sum()
            return loss, loss
        elif self.has_mode('endorsement'):
            if self.has_mode('pretrain'):
                pe = net_output[0].float()
                ne = net_output[1]['neg_out'][0].float()
                p_mask = net_output[1]['p_mask']
                n_mask = net_output[1]['n_mask']
                n_in_p_mask = net_output[1]['n_in_p_mask'].float()
                ploss = (-F.logsigmoid(pe).masked_select(p_mask)).sum()
                nloss = (-F.logsigmoid(-ne*(1 - n_in_p_mask)).masked_select(n_mask)).sum()

                wloss = net_output[1]['wloss'].float().sum() * 0.05
                freq_loss = net_output[1]['freq_loss'].float().sum() * 1

                nll_loss = loss = ploss + nloss + wloss + freq_loss
                if self.has_mode('self_align'):
                    wloss = net_output[1]['self_align_decoder_out'][1]['wloss'].float().sum() * 0.05
                    freq_loss = net_output[1]['self_align_decoder_out'][1]['freq_loss'].float().sum() * 1
                    ploss = net_output[1]['self_align_decoder_out'][1]['ploss']
                    nloss = net_output[1]['self_align_decoder_out'][1]['nloss']
                    align_loss = ploss + nloss + freq_loss
                    loss = nll_loss = loss + align_loss
                return loss, nll_loss
            elif self.has_mode('drop_words'):
                if model.training:
                    sample['target'] = net_output[1]['drop_words_target']
                loss, nll_loss = super().compute_loss(model, net_output, sample)
                return loss, nll_loss
            else:
                if model.training:
                    m = net_output[1]['edm_decoder_out'][1]['m'].detach()
                    s_mask_inf = net_output[1]['edm_decoder_out'][1]['s_mask_inf'].detach()
                    max_match = (m + s_mask_inf.unsqueeze(-1)).max(1)[0]
                    a, b = (float(self.user_mode['a']), float(self.user_mode['b'])) if 'a' in self.user_mode else (1.0, 0.0)
                    weight = (a * (max_match - b)).sigmoid().float()
                    #weight = (max_match).sigmoid().float()
                    #weight[:] = 1
                    # dbg = model.decoder.get_align(sample['net_input']['src_tokens'], sample['target'], net_output[1]['edm_decoder_out'][1])
                    if self.has_mode('dbg_log_endorsement'):
                        data = {'src_tokens' : model.decoder.get_src_words(sample['net_input']['src_tokens'], ''), 'target' : model.decoder.get_src_words(sample['target'], ''), 'attn' : net_output[1]['attn'].data.cpu(), 'm' : m.data.cpu()}
                        torch.save(data, open("output/handcraft/dbg_%s.pt"%model.encoder.task.args.user_mode, "wb"))
                    if self.has_mode('add_exact'):
                        word_exactness = net_output[1]['word_exactness']
                        exactness = word_exactness[sample['target']]
                        tgt_notin_src = 1 - model.decoder.a_in_b(sample['target'], sample['net_input']['src_tokens'])
                        weight_exact = 1 - tgt_notin_src.float() * exactness
                        weight = weight * weight_exact
                    if self.has_mode('hard_weight'):
                        weight = (weight > torch.rand_like(weight)).float()

                    loss, nll_loss = self.get_elementwise_loss(model, net_output, sample)
                    target_mask = sample['target'].ne(self.padding_idx)
                    loss = weight.masked_select(target_mask) * loss.masked_select(target_mask.view(-1))
                    if self.has_mode('sent_weight'):
                        sweight = (((weight * target_mask.float()).sum(1) / target_mask.float().sum(1)).unsqueeze(1) * target_mask.float()).masked_select(target_mask)
                        loss = loss * sweight
                    loss = loss.sum()
                    nll_loss = (weight.masked_select(target_mask) * nll_loss.masked_select(target_mask.view(-1))).sum()
                    #loss = nll_loss = loss.masked_select(target_mask.view(-1)).sum()
                    #loss, nll_loss = super().compute_loss(model, net_output, sample)
                    return loss, nll_loss
                else:
                    pass            
        elif self.is_mode('endorsement', 'bak'):
            pe = net_output[0].float()
            ne = net_output[1]['neg_out'][0].float()
            p_mask = net_output[1]['p_mask']
            n_mask = net_output[1]['n_mask']
            #loss = ((1 + torch.exp(-pe)).log() * p_mask).sum() + ((1 + torch.exp(ne)).log() * n_mask).sum()
            loss = (-F.logsigmoid(pe).masked_select(p_mask)).sum() + (-F.logsigmoid(-ne).masked_select(n_mask)).sum()
            return loss, loss
        elif self.is_mode('attn_endorse'):
            weight = net_output[1]['attn'].max(2)[0].detach().float()#.pow(0.5)
            loss, nll_loss = self.get_elementwise_loss(model, net_output, sample)
            target_mask = sample['target'].ne(self.padding_idx)
            loss = (weight.masked_select(target_mask) * loss.masked_select(target_mask.view(-1))).sum()
            nll_loss = (weight.masked_select(target_mask) * nll_loss.masked_select(target_mask.view(-1))).sum()
            return loss, nll_loss

        loss, nll_loss = super().compute_loss(model, net_output, sample)
        if self.is_mode('rl_word'):
            rl_loss = net_output[1]['encoder_out']['rl_loss'].sum()
            loss = loss + rl_loss
        elif self.has_mode('sep_lm', 'sep_lm1', 'sep_lm2', 'sep_lm3'):
            lm_loss, _ = super().compute_loss(model, net_output[1]['lm_out'], sample)
            if self.has_mode('only_lm'):
                loss = lm_loss
            else:
                loss = loss + lm_loss
        elif self.is_mode('ignore_batch'):
            if model.batch_id % 10 == 1:
                loss, nll_loss = loss * 0, nll_loss * 0
        elif self.is_mode('gated'):
            #gs = torch.stack([x.g.squeeze(-1) for x in net_output[1]['inner_states'][1:]])
            gs = net_output[1]['inner_states'][1].g.squeeze(-1)
            lg = gs.float().sum()
            #print(loss.data.item(), lg.data.item())
            loss = loss + float(self.user_mode['gated']) * lg
        elif self.has_mode('add_lm'):
            lm_loss, lm_nll_loss = super().compute_loss(model, net_output[1]['lm_out'], sample)
            loss = loss + lm_loss
        elif self.is_mode('decomposable1'):
            gs = net_output[1]['inner_states'][-1].g.squeeze(-1)
            lg = gs.float().sum()
            loss = loss + float(self.user_mode['decomposable']) * lg
        elif self.is_mode('rl_edm'):
            if self.training:
                scores = net_output[1]['word_scores'].float()
                lprob = net_output[1]['word_lprob']
                tokens = net_output[1]['word_tokens']
                mask = tokens.ne(model.decoder.pad).float()
                b = 2.0
                #idx = (tokens == model.decoder.model.decoder.dictionary.indices['town'])
                #lp = lprob[idx]
                #rl_loss = -(-1 * lp).sum() * 100
                #rl_loss = -(torch.min(scores - b, 0.1*(scores - b)) * mask * lprob ).sum()
                #2 * sigmoid(0.5 * x - 2) - 1 from -6 to 19
                #rl_loss = -(((scores * 0.5 - 2).sigmoid() * 2 - 1) * mask * lprob ).sum()
                mask = ((scores < b) & tokens.ne(model.decoder.pad))
                rl_loss = -((scores[mask] - b) * lprob[mask] ).sum() * 0.01
                loss = loss + rl_loss
        if self.has_mode('dbg_log_endorsement'):
            data = {'src_tokens' : model.decoder.get_src_words(sample['net_input']['src_tokens'], ''), 'target' : model.decoder.get_src_words(sample['target'], ''), 'attn' : net_output[1]['attn'].data.cpu()}
            torch.save(data, open("output/handcraft/dbg_%s.pt"%model.encoder.task.args.user_mode, "wb"))
        return loss, nll_loss

    def get_log_probs(self, net_output, model):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        return lprobs

    def sub_sample_by_id(self, sample, idx, model):
        new_sample = {}
        for k in ['id', 'target']:
            new_sample[k] = sample[k][idx]
        new_sample['net_input'] = {}
        for k in ['prev_output_tokens', 'src_lengths', 'src_tokens']:
            new_sample['net_input'][k] = sample['net_input'][k][idx]
        new_sample['ntokens'] = new_sample['target'].ne(model.decoder.pad).sum()
        new_sample['nsentences'] = len(idx)
        return new_sample

    def get_elementwise_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=None, reduce=False,
        )
        return loss, nll_loss

    