---
layout: post
title:  "[Boostcamp AI Tech] 7주차 - Huggingface 스터디"
subtitle:   "Naver Boostcamp AI Tech Level2 U Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level2-U-Stage]
use_math: true
---

# 부스트캠프 7주차

## 🤗 [Huggingface Tutorial](https://huggingface.co/course/chapter1)

### [1] Transformer models

* Transformers
  * 특징
    * 모든 Transformer 모델들은 *language model* 로 학습됨
    * *transfer learning* 과정을 통해 주어진 작업에 맞게 *fine-tuning* 하기
    * 편견과 한계 : 원본 모델이 성차별, 인종차별 또는 동성애 혐오 콘텐츠를 매우 쉽게 생성할 수 있다. fine-tuning을 하더라도 이러한 내재적 편향이 사라지지 않는다.
  * 주요 구성요소
    * Encoder : 모델이 입력으로부터 이해를 얻도록 최적화
    * Decoder : Encoder의 표현을 다른 입력과 함께 사용하여 출력 생성에 최적화
  * **BERT-like** \| *auto-encoding* models \| Encoder-only models
    * 종류 : [ALBERT](https://huggingface.co/transformers/model_doc/albert.html) \| [BERT](https://huggingface.co/transformers/model_doc/bert.html) \| [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) \| [ELECTRA](https://huggingface.co/transformers/model_doc/electra.html) \| [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)
    * 특징 : 문장 분류(Sentence Classification) 및 명명된 개체 인식(Named Entity Recognition), 추출 질문 답변(Extractive Question Answering)과 같이 입력에 대한 이해가 필요한 작업에 적합 (bi-directional attention)
  * **GPT-like** \| *auto-regressive* models \| Decoder-only models
    * 종류 : [CTRL](https://huggingface.co/transformers/model_doc/ctrl.html) \| [GPT](https://huggingface.co/transformers/model_doc/gpt.html) \| [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html) \| [Transformer XL](https://huggingface.co/transformers/model_doc/transformerxl.html)
    * 특징 : 텍스트 생성(Text Generation)과 같은 생성 작업에 적합
  * **BART/T5-like** \| *sequence-to-sequence* models \| Encoder-Decoder models
    * 종류 : [BART](https://huggingface.co/transformers/model_doc/bart.html) \| [mBART](https://huggingface.co/transformers/model_doc/mbart.html) \| [Marian](https://huggingface.co/transformers/model_doc/marian.html) \| [T5](https://huggingface.co/transformers/model_doc/t5.html)
    * 특징 : 요약(Summarization), 번역(Translation) 또는 생성적 질문 답변(Generative Question Answering)과 같이 주어진 입력에 따라 새로운 문장을 생성하는 작업에 가장 적합

* High-Level API
  * [pipeline](https://huggingface.co/transformers/main_classes/pipelines.html) : 모델을 전처리부터 후처리까지 연결하여 간편하게 답을 얻을 수 있음

    ```python
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    classifier("I've been waiting for a HuggingFace course my whole life.")
    ```

  * [Inference API](https://huggingface.co/inference-api) : 브라우저를 통해 직접 테스트 가능

### [2] Using huggingface transformers

1. Preprocessing with a tokenizer
   * Tokenizer : 텍스트 입력을 모델이 이해할 수 있는 숫자로 변환함
     * 입력을 *토큰* 이라고 하는 단어, 하위 단어 또는 기호로 분할
     * 각 토큰을 정수로 매핑
     * 모델에 유용할 수 있는 추가 입력 추가
     ```python
     from transformers import AutoTokenizer
     raw_inputs = [
         "I've been waiting for a HuggingFace course my whole life.", 
         "I hate this so much!",
     ]
     checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
     inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
     print(inputs) # 'input_ids', 'attention_mask'
     ```
   * 종류
     * Word-based
     * Character-based
     * Subword tokenization
   * 과정
     * Encoding
     * Decoding

2. Going through the model
   * Model head : hidden states의 고차원 벡터를 입력으로 받아 다른 차원에 투영
     ![image](https://user-images.githubusercontent.com/35680202/134797641-198ee928-542a-4447-b923-471881e61972.png)
     ```python
     from transformers import AutoModel, AutoModelForSequenceClassification
     checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
     #model = AutoModel.from_pretrained(checkpoint)
     model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
     outputs = model(**inputs)
     ```
     * `*Model` : retrieve the hidden states
     * `*ForSequenceClassification` : model with a sequence classification head (to be able to classify the sentences as positive or negative)
   * Different loading methods
     ```python
     from transformers import BertConfig, BertModel
     # random initialized model
     config = BertConfig()
     model = BertModel(config)
     # pre-trained model
     model = BertModel.from_pretrained("bert-base-cased") # ~/.cache/huggingface/transformers 에 다운
     ```
   * Saving methods
     ```python
     model.save_pretrained(PATH) # config.json, pytorch_model.bin 파일에 저장됨
     ```
     * `config.json` : attributes necessary to build the model architecture
     * `pytorch_model.bin` : *state dictionary*, model’s weights

3. Postprocessing the output
   * 모델에서 출력으로 얻은 값이 반드시 그 자체로 의미가 있는 것은 아니다.
   * 일반적으로 SoftMax 레이어를 통과한 후 교차 엔트로피와 같은 실제 손실 함수에 넣는다.
   * `model.config.id2label`를 이용해서 사람이 이해할 수 있는 라벨로 변환할 수 있다.
   ```python
   import torch
   predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
   print(predictions)
   ```

## 🤗 [Huggingface Notebooks](https://huggingface.co/transformers/notebooks.html)

1. How to fine-tune a model on text classification
2. How to fine-tune a model on language modeling
3. How to fine-tune a model on token classification
4. How to fine-tune a model on question answering
5. How to fine-tune a model on multiple choice
6. How to fine-tune a model on translation
7. How to fine-tune a model on summarization

* [huggingface datasets viewer](https://huggingface.co/datasets/viewer/)
