---
layout: post
title:  "[Boostcamp AI Tech] 9주차 - KLUE (이론 강의)"
subtitle:   "Naver Boostcamp AI Tech Level2 P Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level2-P-Stage]
use_math: true
---

# 부스트캠프 9주차

## [1강] 인공지능과 자연어 처리

* 인공지능의 탄생과 자연어 처리
  * [ELIZA 챗봇](https://www.eclecticenergies.com/psyche/eliza) : 최초의 대화형(chitchat) 챗봇, 튜링 테스트를 적용할 수 있는 최초의 Human-Like AI
  * 컴퓨터의 자연어 처리
    * Encoder : 벡터 형태로 자연어를 인코딩
    * Decoder : 벡터를 자연어로 디코딩
  * 자연어 단어 임베딩
    * Word2Vec : 중심 단어의 주변 단어들을 이용해 중심 단어를 추론하는 방식으로 학습
      * 장점 : 벡터 연산을 통한 추론이 가능
      * 단점 : subword, OOV(Out of vocabulary) 에서 적용 불가능
    * [FastText](https://research.fb.com/fasttext/) : Word2Vec과 유사하지만, n-gram으로 나누어 학습
      * [[Paper Reivew] FastText: Enriching Word Vectors with Subword Information](https://www.youtube.com/watch?v=7UA21vg4kKE)]
      * 장점 : 오탈자, OOV, 등장 횟수가 적은 학습 단어에 강세
      * 단점 : 동형어, 다의어 문제나 문맥을 고려하지 못하는 것은 여전함
* 딥러닝 기반의 자연어 처리와 언어모델
  * 언어모델 : 자연어의 법칙을 컴퓨터로 모사한 모델, 주어진 단어들로부터 그 다음에 등장할 단어의 확률을 예측하는 방식으로 학습
    * 마코트 체인 모델(Markov Chain Model) : 통계와 단어의 n-gram을 기반으로 계산
    * RNN 기반의 언어모델 : 최종 출력된 context vector 를 이용하여 분류하는 등의 방식
  * [Seq2Seq](https://www.youtube.com/watch?v=4DzKM0vgG1Y)
    * RNN 기반의 Seq2Seq : RNN 구조인 Encoder를 통해 얻은 context vector 를 RNN 구조인 Decoder에 넣어 출력
  * [Seq2Seq + Attention](https://www.youtube.com/watch?v=WsQLdu2JMgI)
    * RNN 기반의 Seq2Seq with Attention : 긴 입력 시퀀스, 고정된 context vector 등 RNN 구조의 문제점을 해결하기 위해 등장
  * **Selt-Attention**
    * Transformer : 다양한 언어모델들이 Transformer를 기반으로 발전하고 있다.

* Further Questions
  * Embedding이 잘 되었는지, 안되었는지를 평가할 수 있는 방법은 무엇이 있을까요?
    - WordSim353, Spearman's correlation, Analogy test
  * Vanilar Transformer는 어떤 문제가 있고, 이걸 어떻게 극복할 수 있을까요?
    - Longformer, Linformer, Reformer

## [2강] 자연어의 전처리

* 통계학적 분석
  * Token 개수 파악 후 아웃라이어 제거
  * 빈도수 확인 후 사전(dictionary) 정의
* **전처리(Preprocessing)**
  * 기능
    * 학습에 사용될 데이터를 수집&가공하는 모든 프로세스
    * Task의 성능을 가장 확실하게 올릴 수 있는 방법
  * 종류
    * 개행문자, 공백 제거 + 띄어쓰기, 문장분리 보정
    * 특수문자, 이메일, 링크, 불용어, 조사, 제목, 중복 표현 제거
  * 문자열 함수
    * 대소문자 변환 : `upper()`, `lower()`, `capitalize()`, `title()`, `swapcase()`
    * 편집, 치환 : `strip()`, `rstrip()`, `lstrip()`, `replace(a, b)`
    * 분리, 결합 : `split()`, `''.join(list)`, `lines.splitlines()`
    * 구성 문자열 판별 : `isdigit()`, `isalpha()`, `isalnum()`, `islower()`, `isupper()`, `isspace()`, `startswith('hi')`, `endswith('hi')`
    * 검색 : `count('hi')`, `find('hi')`, `rfind('hi')`, `index('hi')`, `rindex('hi')`
* **토큰화(Tokenizing)**
  * 기능
    * 주어진 데이터를 토큰(Token) 단위로 나누는 작업
    * 어절, 단어, 형태소, 음절, 자소, WordPiece 등
  * 종류 예시
    * 문장 토큰화(Sentene Tokenizing) : 문장 분리
    * 단어 토큰화(Word Tokenizing) : 구두점 분리, 단어 분리
  * 한국어 토큰화
    * 영어와 달리 띄어쓰기만으로는 부족
    * 따라서 어절을 의미를 가지는 최소 단위인 형태소로 분리함

## [3강] BERT 언어모델 소개

* BERT 언어모델
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  * [BERT 톺아보기](http://docs.likejazz.com/bert/)
  * 모델 구조 : Transformer Encoder, All-to-all network
  * 사전학습 태스크 : Masked Language Model, Next Sentence Prediction
* BERT 모델 응용 (`BERT Multi-lingual pretrained model` 사용)
  * 감성 분석 데이터셋 : [네이버 영화 리뷰 코퍼스](https://github.com/e9t/nsmc)
  * 관계 추출 데이터셋 : KAIST가 구축한 Silver data
  * 의미 비교 데이터셋 : 디지털 동반자 패러프레이징 질의 문장 데이터를 이용하여 질문-질문 데이터 생성 및 학습
  * 개체명 분석 : ETRI 개체명 인식 데이터
  * 기계 독해 : LG CNS가 공개한 한국어 AQ 데이터셋, [KorQuAD](https://korquad.github.io/)
* 한국어 BERT 모델
  * [한국어 tokenizing에 따른 성능 비교](https://arxiv.org/abs/2010.02534)
  * Advanced BERT model : KBQA에서 주요 entity 추출, entity tag 부착, entity embedding layer 추가를 통해 개선 가능

## [4강] 한국어 BERT 언어 모델 학습

* BERT 모델 학습 : 도메인 특화 Task의 경우, 도메인 특화 학습 데이터를 사용하여 새로 학습하는 것이 좋다.
* BERT 모델 학습 단계
  1. Tokenizer 만들기
  2. 데이터셋 확보
  3. Next Sentence Prediction (NSP)
  4. Masked Language Model (MLM)
* 학습을 위한 데이터
  * `input_ids` : for Token Embeddings
  * `token_type_ids` : Segment Embeddings
* [BERT 추가 설명](https://jiho-ml.com/weekly-nlp-28/)

## [5강] BERT 기반 단일 문장 분류 모델 학습

* KLUE 데이터셋 소개
  * KLUE 데이터셋 : 한국어 자연어 이해 벤치마크 (Korean Language Understanding Evaluation, KLUE)
  * 단일 문장 분류 task : 문장 분류, 관계 추출
  * 문장 임베딩 벡터의 유사도 : 문장 유사도
  * 두 문장 관계 문류 task : 자연어 추론
  * 문장 토큰 분류 task : 개체명 인식, 품사 태깅, 질의 응답
  * DST : 목적형 대화
  * 의존 구문 분석 : 단어들 사이의 관계를 분석하는 task (의존소, 지배소) → 복잡한 자연어 형태를 그래프로 구조화해서 표현 가능
* 단일 문장 분류 task 소개
  * 문장 분류 task : 주어진 문장이 어떤 종류의 범주에 속하는지를 구분하는 task
    * 감성분석(Sentiment Analysis) : 문장의 긍정 또는 부정 및 중립 등 성향을 분류하는 프로세스 → 혐오 발언 분류, 기업 모니터링 등
    * 주제 라벨링(Topic Labeling) : 문장의 내용을 이해하고 적절한 범주를 분류하는 프로세스 → 대용량 문서 분류, VoC(Voice of Customer) 고객의 피드백 주제 분류
    * 언어감지(Language Detection) : 문장이 어떤 나라 언어인지를 분류하는 프로세스 → 번역기, 데이터 필터링
    * 의도 분류(Intent Classification) : 문장이 가진 의도를 분류하는 프로세스 → 챗봇
  * 문장 분류를 위한 데이터
    * Kor_hate : 혐오 표현에 대한 데이터
    * Kor_sarcasm : 비꼬는/비꼬지 않은 표현의 문장
    * Kor_sae : 질문 종류/명령 종류
    * Kor_3i4k : 단어 또는 문장 조각
* 단일 문장 분류 모델 학습
  * 모델 구조 : BERT의 `[CLS]` token의 vector를 classification 하는 Dense layer 사용
  * 주요 매개변수
    * `input_ids` : sequence token을 입력
    * `attention_mask` : \[0, 1\]로 구성된 마스크, 패딩 토큰 구분
    * `token_type_ids` : \[0, 1\]로 구성되었으며 입력의 첫 문장과 두번째 문장 구분
    * `position_ids` : 각 입력 시퀀스의 임베딩 인덱스
    * `inputs_embeds` : input_ids 대신 직접 임베딩 표현을 할당
    * `labels` : loss 계산을 위한 레이블
    * `next_sentence_label` : 다음 문장 예측 loss 계산을 위한 레이블
  * 학습 과정
    * 데이터 : Dataset 다운로드, Dataset 전처리 및 토큰화, Dataloader 설계, Train, Test Dataset 준비
    * 학습 : TrainingArguments 설정, Pretrained Model import, Trainer 설정, Model 학습
    * 추론 : Predict 함수 구현 및 평가

## [6강] BERT 기반 두 문장 관계 분류 모델 학습

* 두 문장 관계 분류 task 소개
  * 두 문장 관계 분류 task : 주어진 2개의 문장에 대해, 두 문장의 자연어 추론과 의미론적인 유사성을 측정하는 task
  * 두 문장 관계 분류를 위한 데이터
    * Natural Language Inference (NLI)
      * 언어모델이 자연어의 맥락을 이해할 수 있는지 검증하는 task
      * 전체문장(Premise), 가설문장(Hypothesis), 함의(Entailment), 모순(Contradiction), 중립(Neutral) 으로 분류
    * Semantic text pair : 두 문장의 의미가 서로 같은 문장인지 검증하는 task
* 두 문장 관계 분류 모델 학습
  * Information Retrieval Question and Answering (IRQA)

## [7강] BERT 언어모델 기반의 문장 토큰 분류

* 문장 토큰 분류 task 소개
  * 문장 토큰 분류 task : 주어진 문장의 각 token이 어떤 범주에 속하는지 분류하는 task
    * Named Entity Recognition (NER) : 개체명 인식, 문맥을 파악해서 인명, 기관명, 지명 등과 같은 문장 또는 문서에서 특정한 의미를 가지고 있는 단어 또는 어구 등을 인식하는 과정
    * Part-of-speech tagging (POS Tagging) : 주어진 문장의 각 성분에 대하여 가장 알맞은 품사를 태깅하는 것
  * 문장 토큰 분류를 위한 데이터
    * kor_ner : 한국어해양대학교 자연어 처리 연구실에서 공개한 한국어 NER 데이터셋
* 문장 토큰 분류 모델 학습
  * NER fine-tuning with BERT : 형태소 단위 토큰을 **음절 단위**의 토큰으로 분해하고, Entity tag 역시 음절 단위로 매핑시켜 주어야 한다.
