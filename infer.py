import argparse
import functools
import time
import warnings

from src.predict import SRPredictor
from src.utils.utils import add_arguments, print_arguments
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/deepspeech2.yml',     "配置文件")
add_arg('wav_path',         str,    'test.wav',          "预测音频的路径")
add_arg('use_gpu',          bool,   False,                        "是否使用GPU预测")
add_arg('is_itn',           bool,   False,                       "是否对文本进行反标准化")
add_arg('model_path',       str,    'models/deepspeech2_non-streaming_fbank/inference.pt', "导出的预测模型文件路径")
args = parser.parse_args()
print_arguments(args=args)


# 获取识别器
predictor = SRPredictor(configs=args.configs,
                        model_path=args.model_path,
                        use_gpu=args.use_gpu)


# 短语音识别
def predict_audio():
    start = time.time()
    result = predictor.predict(audio_data=args.wav_path, is_itn=args.is_itn)
    score, text = result['score'], result['text']
    print(f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {text}, 得分: {int(score)}")


if __name__ == "__main__":
    predict_audio()
