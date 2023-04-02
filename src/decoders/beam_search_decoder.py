import os

from src.decoders.swig_wrapper import (
    Scorer,
    CTCBeamSearchDecoder,
    ctc_beam_search_decoding_batch,
    ctc_beam_search_decoding
)


class BeamSearchDecoder:
    def __init__(self, alpha, beta, beam_size, cutoff_prob, cutoff_top_n, vocab_list, num_processes=10,
                 blank_id=0, language_model_path='lm/zh_giga.no_cna_cmn.prune01244.klm'):
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.vocab_list = vocab_list
        self.num_processes = num_processes
        self.blank_id = blank_id
        if not os.path.exists(language_model_path) and language_model_path == 'lm/zh_giga.no_cna_cmn.prune01244.klm':
            print('=' * 70)
            language_model_url = 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
            print("语言模型不存在，正在下载，下载地址： %s ..." % language_model_url)
            os.makedirs(os.path.dirname(language_model_path), exist_ok=True)
            os.system("wget -c " + language_model_url + " -O " + language_model_path)
            print('=' * 70)
        print('=' * 70)
        print("初始化解码器...")
        self._ext_scorer = Scorer(alpha, beta, language_model_path, vocab_list)
        lm_char_based = self._ext_scorer.is_character_based()
        lm_max_order = self._ext_scorer.get_max_order()
        lm_dict_size = self._ext_scorer.get_dict_size()
        print(f"language model: "
              f"model path = {language_model_path}, "
              f"is_character_based = {lm_char_based}, "
              f"max_order = {lm_max_order}, "
              f"dict_size = {lm_dict_size}")
        batch_size = 1
        self.beam_search_decoder = CTCBeamSearchDecoder(vocab_list, batch_size, beam_size, num_processes, cutoff_prob,
                                                        cutoff_top_n, self._ext_scorer, self.blank_id)
        print("初始化解码器完成!")
        print('=' * 70)

    # 单个数据解码
    def decode_beam_search_offline(self, probs_seq):
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(self.alpha, self.beta)
        # beam search decode
        beam_search_result = ctc_beam_search_decoding(probs_seq=probs_seq,
                                                      vocabulary=self.vocab_list,
                                                      beam_size=self.beam_size,
                                                      ext_scoring_func=self._ext_scorer,
                                                      cutoff_prob=self.cutoff_prob,
                                                      cutoff_top_n=self.cutoff_top_n,
                                                      blank_id=self.blank_id)
        return beam_search_result[0]

    # 一批数据解码
    def decode_batch_beam_search_offline(self, probs_seqs):
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(self.alpha, self.beta)
        # beam search decode
        self.num_processes = min(self.num_processes, len(probs_seqs))
        beam_search_results = ctc_beam_search_decoding_batch(probs_seqs=probs_seqs,
                                                             vocabulary=self.vocab_list,
                                                             beam_size=self.beam_size,
                                                             num_processes=self.num_processes,
                                                             ext_scoring_func=self._ext_scorer,
                                                             cutoff_prob=self.cutoff_prob,
                                                             cutoff_top_n=self.cutoff_top_n,
                                                             blank_id=self.blank_id)
        results = [result[0][1] for result in beam_search_results]
        return results
