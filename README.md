# Image Captioning On SageMaker
An image captioner built on Amazon SageMaker.

## Overview
This model uses a pre-trained Resnet-50 network with a fine-tuned final embedding layer as an encoder and a [transformer decoder](https://arxiv.org/abs/1706.03762) as the decoder.
Downloading and preprocessing of data, training, and inference is all done via SageMaker Studio.

## Environment
- Training was run remotely using SageMaker's PyTorch 2.0.0 Python 3.10 GPU optimized image on an ml.g4dn.xlarge instance.
- Notebook, inference, and preprocessing were run on the same ml.t3.medium instance using SageMaker's PyTorch 2.0.0 Python 3.10 CPU optimized image.

## Requirements
- Preprocessing requires the spacy python module and embeddings. Installed via a lifecycle configuration [script](https://github.com/JohnRB626/ImageCaptioningOnSageMaker/blob/main/on-kernel-start.sh).
- Training requires nltk module for bleu score calculation. This is provided to estimator by `requirements.txt`.

## Instructions
1. Clone repository into SageMaker Studio
2. Download data using `get_coco.sh`. The default sagemaker bucket should be passed as argument
3. Run `python text_preprocess.py all train2017` followed by `python text_preprocess.py all val2017`
4. Run cells in `ImageCaption.ipynb`

## Results
![Figure: 1](https://github.com/JohnRB626/ImageCaptioningOnSageMaker/blob/main/results/figure01.png)
![Figure: 2](https://github.com/JohnRB626/ImageCaptioningOnSageMaker/blob/main/results/figure02.png)
![Figure: 3](https://github.com/JohnRB626/ImageCaptioningOnSageMaker/blob/main/results/figure03.png)
![Figure: 4](https://github.com/JohnRB626/ImageCaptioningOnSageMaker/blob/main/results/figure04.png)
![Figure: 5](https://github.com/JohnRB626/ImageCaptioningOnSageMaker/blob/main/results/figure05.png)
