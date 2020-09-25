from fairseq.tasks.translation import TranslationTask
from . import register_task

@register_task('nlg')
class NLGTask(TranslationTask):
    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            #from fairseq.sequence_generator import SequenceGenerator # maple
            from fairseq.models.sequence_generator_nlg import NLGSequenceGenerator
            return NLGSequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )