## Introduction

基于Pytorch的DeepSpeech2语音识别模型实现

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* yaml
* cn2an
* termcolor
* Levenshtein
* typeguard
* paddlespeech_ctcdecoders(仅限linux平台)
* ffmpeg
* resampy
* scipy
* pydub
* zhconv
* torchaudio
* pillow

## DataSet

数据集为aishell,178小时的中文语音数据集.
语音数据[aishell](https://openslr.magicdatatech.com/resources/33/data_aishell.tgz)
, 噪声数据[noise](http://www.openslr.org/resources/28/rirs_noises.zip), 如果不需要数据增强操作可以不用噪声数据.
下载完成后将两个压缩文件放至`data`文件夹下`

## Train

* 数据预处理, 分别对语音和噪声数据进行解压缩操作, 之后运行`create_data.py`进行数据预处理操作.
  模型配置文件为`config/deepspeech2.yml`, 数据扩充配置文件为`config/augmentation.json`

```
    cd ./data_preprocess
    python aishell.py
    python noise.py
    cd ..
    python create_data.py
```

* 训练模型

```
    python train.py
```

* 导出模型

```
    python export_model.py
```

* 模型预测

```
    python infer.py
```

## Results

|             | Val_cer | Test_cer |
|:------------|:--------|:---------|
| DeepSpeech2 | 0.08212 | 0.0936   |

cer表示字错率, 即预测文本与真实文本之间的编辑距离

## Analysis

本项目主要针对[MASR](https://github.com/yeyupiaoling/MASR)项目的复现, 只抽取其中的语音识别部分进行复现. 简化了部分流程方便阅读,
通过此项目, 了解了语音识别的整个流程, 包括数据预处理, 数据扩充, 模型搭建, CTC损失, 模型评估, 并进行了基本实现. 其中在CTC上遇到小问题, windows平台上的`beam search`效果很差, 找了很多实现结果均不如`greedy`. 总体学习到新的项目代码架构, 了解部分python库的使用. 收货良多.

## Reference

[1] [MASR](https://github.com/yeyupiaoling/MASR)
