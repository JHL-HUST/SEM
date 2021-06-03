## Synonyms Encoding Method (SEM)

This repository contains necessary code for reproducing main results in the paper:

[**Natural Language Adversarial Defense through Synonym Encoding**](https://arxiv.org/abs/1909.06723) (UAI 2021)

[Xiaosen Wang](https://xiaosen-wang.github.io/), Hao Jin, Yichen Yang and Kun He

**For IGA attack, please refer to [IGA](https://github.com/xiaosen-wang/SEM)!**

## Datesets
There are three datasets used in our experiments:

- [IMDB](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [Yahoo! Answers](https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz)

## Requirements
The code was tested with:

- python 3.6.5
- numpy 1.16.4
- tensorflow 1.8.0
- tensorflow-gpu 1.5.0
- pandas 0.23.0
- keras 2.2.0
- scikit-learn 0.19.1
- scipy 1.0.1

## File Description

- `textrnn.py`,`textcnn.py`,`textbirnn.py` : The models for LSTM, Word-CNN and Bi-LSTM.
- `train_orig.py`,`train_enc.py`: Training models with or without SEM.
- `glove_utils.py` : Loading the glove model and create embedding matrix for word dictionary.
- `build_embeddings.py` : Generating the embedding matrix for original word dictionary and encoded word dictionary

## Experiments

1. Generating the embedding matrix for original dictionary and encoded dictionary:

    ```shell
    python build_embedding.py
    ```

2. Training the models with the original word dictionary:

    ```shell
    python train_orig.py --data aclImdb --sn 10 --sigma 0.5 --nn_type textrnn
    ```

3. Training the models with the encoded word dictionary:

    ```shell
    python train_enc.py --data aclImdb --sn 10 --sigma 0.5 --nn_type textrnn
    ```

## Contact

This repository is under active development. Questions and suggestions can be sent to xswanghuster@gmail.com.