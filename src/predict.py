import os
import platform

import cn2an
import numpy as np
import yaml

from src.data_utils.audio import AudioSegment
from src.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from src.data_utils.featurizer.text_featurizer import TextFeaturizer
from src.decoders.ctc_greedy_decoder import greedy_decoder
from src.infer_utils.inference_predictor import InferencePredictor
from src.utils.logger import setup_logger
from src.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class SRPredictor:
    def __init__(self,
                 configs=None,
                 model_path=None,
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置文件路径
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if not isinstance(configs, str) or not os.path.exists(configs):
            raise ValueError('configs文件不存在')
        with open(configs, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        print_arguments(configs=configs)

        self.configs = dict_to_object(configs)
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self._text_featurizer = TextFeaturizer(vocab_filepath=self.configs.dataset_conf.dataset_vocab)
        self._audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        self.__init_decoder()
        # 创建模型
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件{model_path}不存在，！")
        # 获取预测器
        self.predictor = InferencePredictor(configs=self.configs,
                                            use_model=self.configs.use_model,
                                            streaming=self.configs.streaming,
                                            model_path=model_path,
                                            use_gpu=self.use_gpu)
        # 预热
        for _ in range(5):
            warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(134240,))
            self.predict(audio_data=warmup_audio, is_itn=False)

    # 初始化解码器
    def __init_decoder(self):
        # 集束搜索方法的处理
        if self.configs.decoder == "ctc_beam_search":
            if platform.system() != 'Windows':
                try:
                    from src.decoders.beam_search_decoder import BeamSearchDecoder
                    self.beam_search_decoder = BeamSearchDecoder(vocab_list=self._text_featurizer.vocab_list,
                                                                 **self.configs.ctc_beam_search_decoder_conf)
                except ModuleNotFoundError:
                    logger.warning('==================================================================')
                    logger.warning('缺少 paddlespeech-ctcdecoders 库，请根据文档安装。')
                    logger.warning('【注意】已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                    logger.warning('==================================================================\n')
                    self.configs.decoder = 'ctc_greedy'
            else:
                logger.warning('==================================================================')
                logger.warning(
                    '【注意】Windows不支持ctc_beam_search，已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.configs.decoder = 'ctc_greedy'

    # 解码模型输出结果
    def decode(self, output_data, is_itn):
        """
        解码模型输出结果
        :param output_data: 模型输出结果
        :param is_itn: 是否对文本进行反标准化
        :return:
        """
        # 执行解码
        if self.configs.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            result = self.beam_search_decoder.decode_beam_search_offline(probs_seq=output_data)
        else:
            # 贪心解码策略
            result = greedy_decoder(probs_seq=output_data, vocabulary=self._text_featurizer.vocab_list)

        score, text = result[0], result[1]
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)
        return score, text

    # 预测音频
    def predict(self,
                audio_data,
                is_itn=False,
                sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        audio_feature = self._audio_featurizer.featurize(audio_segment)
        input_data = np.array(audio_feature).astype(np.float32)[np.newaxis, :]
        audio_len = np.array([input_data.shape[1]]).astype(np.int64)

        # 运行predictor
        output_data = self.predictor.predict(input_data, audio_len)[0]

        # 解码
        score, text = self.decode(output_data=output_data, is_itn=is_itn)
        result = {'text': text, 'score': score}
        return result

    # 对文本进行反标准化
    @staticmethod
    def inverse_text_normalization(text):
        return cn2an.transform(text, "cn2an")
