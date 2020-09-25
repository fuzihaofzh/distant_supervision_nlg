from fairseq.models.transformer import TransformerModel, register_model, TransformerEncoder, TransformerDecoder, register_model_architecture, Linear
from fairseq.models import BaseFairseqModel, FairseqEncoderDecoderModel, FairseqIncrementalDecoder
from undecorated import undecorated
from fairseq.models.transformer import base_architecture as base_architecture_transformer
import torch.nn as nn
import json
import os
import copy
import torch
import sys


MODEL_NAME = "simple_transformer"

@register_model(MODEL_NAME)
class SimpleTransformerModel(BaseFairseqModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--user-mode',
                            type=str,
                            help='user-mode')
        parser.add_argument('--results-dir', type=str, help='results-dir')

    @classmethod
    def build_model(cls, args, task):
        tmodel = TransformerModel.build_model(args, task)
        model = SimpleTransformerModel(tmodel)
        mode = {e.split('=')[0] : e.split('=')[1] if len(e.split('=')) > 1 else None  for e in args.user_mode.split(',')}
        model.user_mode = mode
        model.decoder = ProxyDecoder(tmodel, model.user_mode)
        model.encoder = ProxyEncoder(tmodel, model.user_mode)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def parameters(self, recurse=True):
        return super().parameters(recurse)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.model.get_normalized_probs(net_output, log_probs, sample = sample)

    def max_decoder_positions(self):
        return self.model.decoder.max_positions()

    def evaluate_model_independent(self, task, epoch_itr):

        epoch = epoch_itr.epoch

        command1 = Process(['sh', './scripts/eval_auto.sh', task.args.data.split('/')[-1], task.args.save_dir.split('/')[-1], str(epoch_itr.epoch), str(task.args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command1.register_callback(log_tb_result, command1, task, epoch)
        self.sub_command1 = command1

    def log_tb_result(self, command, task, epoch):
        stdout = command.stdout.read().decode("utf-8") 
        print(stdout)
        print(command.stderr.read().decode("utf-8") )
        result = stdout.strip().split('\n')[-1].replace("'", '"')
        result = json.loads(result)

        if not hasattr(self, "tb_writer"):
            from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(os.path.join(task.args.tensorboard_logdir, "eval"), flush_secs=5)
        for k in result:
            self.tb_writer.add_scalar(k, result[k], epoch)

    def __del__(self): 
        if hasattr(self, "sub_command1"):
            self.sub_command1.wait()
        if hasattr(self, "tb_writer"):
            self.tb_writer.close()

    def evaluate_model(self, task, epoch_itr, trainer):
        torch.cuda.empty_cache()
        args_bak = sys.argv
        log_file = os.path.join(task.args.results_dir, "log.%s.txt"%str(epoch_itr.epoch))
        print("Log saves to: ", log_file)
        sys.argv = args_bak[:1] + ("%s -s %s -t %s --path no_need --beam 5 --max-tokens %d --max-len-b 80 --results-path %s --remove-bpe"%(task.args.data, task.args.source_lang, task.args.target_lang, task.args.max_tokens // 8, log_file)).split()
        parser = options.get_generation_parser()
        args = options.parse_args_and_arch(parser)
        sys.argv = args_bak
        
        if not os.path.exists(task.args.results_dir):
            os.makedirs(task.args.results_dir)

        trainer.model.eval()
        generate_main(args, trainer.model)
        trainer.model.train()

        # make output
        result = sorted([[int(r.split('\t')[0][2:]), r.split('\t')[2]] for r in open(args.results_path, encoding = 'utf-8').readlines()[2::4]])
        output = os.path.join(task.args.results_dir, "output.%s.txt"%str(epoch_itr.epoch))
        open(output, "w", encoding = 'utf-8').write(''.join([r[1] for r in result]))

        # eval
        open("output/eval/{expname}/eval.{epoch}.txt".format(expname = os.path.split(task.args.results_dir)[-1], epoch = epoch_itr.epoch), "w", encoding = "utf-8").write(str(task.args) + '\n\n')
        cmd = "./e2e-metrics/measure_scores.py -p output/preprocessed/{dataset}/test.{tgt} output/eval/{expname}/output.{epoch}.txt 2>&1 | tee -a  output/eval/{expname}/eval.{epoch}.txt".format(dataset = os.path.split(task.args.data)[-1], expname = os.path.split(task.args.results_dir)[-1], epoch = epoch_itr.epoch, tgt = "ref")#task.args.target_lang

        command1 = Process(cmd, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command1.register_callback(self.log_tb_result, command1, task, epoch_itr.epoch)
        self.sub_command1 = command1

    def set_epoch_id(self, epoch_id):
        self.epoch_id = epoch_id

    def set_batch_id(self, batch_id):
        self.batch_id = batch_id


class ProxyModule():# 不要继承nn.Module，不然model会有两份。
    def __init__(self, model, uer_mode, args):
        super(ProxyModule, self).__init__()
        self.model = model
        self.user_mode = uer_mode
        self.args = args
        self.pad = model.encoder.dictionary.pad()
        self.unk = model.encoder.dictionary.unk()
        self.eos = model.encoder.dictionary.eos()

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)

    def is_mode(self, *args):
        return all([m in self.user_mode for m in args])

    def has_mode(self, *args):
        return any([m in self.user_mode for m in args])

    def a_in_b(self, a, b):
        token_mask = a.ne(self.pad)
        tinr = (a.unsqueeze(2) == b.unsqueeze(1)).max(2)[0].long() * token_mask.long()
        return tinr

    def get_src_words(self, src):
        results = []
        for i in range(src.shape[0]):
            results.append(' '.join([self.model.encoder.dictionary.symbols[j] for j in src[i]]).replace("@@ ", ""))
        return results

    def get_tgt_words(self, tgt):
        results = []
        for i in range(tgt.shape[0]):
            results.append(' '.join([self.model.decoder.dictionary.symbols[j] for j in tgt[i]]).replace("@@ ", ""))
        return results

    def get_words(self, tokens):
        return self.get_tgt_words(tokens)

class ProxyEncoder(ProxyModule):    
    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_out = self.model.encoder(src_tokens, src_lengths, **kwargs)
        return encoder_out
    def reorder_encoder_out(self, *args):
        return self.model.encoder.reorder_encoder_out(*args)

class ProxyDecoder(ProxyModule):
    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        return self.model.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)


@register_model_architecture(MODEL_NAME, MODEL_NAME)
def base_architecture_simple_transformer(args):
    undecorated(base_architecture_transformer)(args)

from threading import Thread
import time
import subprocess


class Process(subprocess.Popen):
    def register_callback(self, callback, *args, **kwargs):
        Thread(target=self._poll_completion, args=(callback, args, kwargs)).start()

    def _poll_completion(self, callback, args, kwargs):
        while self.poll() is None:
            time.sleep(0.1)
        callback(*args, **kwargs)

# ====================
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
def generate_main(args, model):
    models = [model]
    result_writer = open(args.results_path, "w", encoding = "utf-8")

    #assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    #print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    #print('| loading model(s) from {}'.format(args.path))
    #models, _model_args = checkpoint_utils.load_model_ensemble(
    #    args.path.split(':'),
    #    arg_overrides=eval(args.model_overrides),
    #    task=task,
    #)

    # Optimize ensemble for generation
    #for model in models:
    #    model.make_generation_fast_(
    #        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
    #        need_attn=args.print_alignment,
    #    )
    #    if args.fp16:
    #        model.half()
    #    if use_cuda:
    #        model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str), file = result_writer)
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str), file = result_writer)

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str), file = result_writer)
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ), file = result_writer)

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ), file = result_writer)

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    return scorer