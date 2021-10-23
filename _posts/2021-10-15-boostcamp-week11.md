---
layout: post
title:  "[Boostcamp AI Tech] 11주차 - MRC (이론 강의)"
subtitle:   "Naver Boostcamp AI Tech Level2 P Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level2-P-Stage]
use_math: true
---

# 부스트캠프 11주차

## [1강] MRC Intro & Python Basics

* 강의 목표 : 사용자의 질문에 답할 수 있는 Question Answering 모델을 밑바닥부터 개발

### Introduction to MRC

* 기계 독해 (Machine Reading Comprehension) : 주어진 지문(Context)를 이해하고, 주어진 질의(Query/Question)의 답변을 추론하는 문제
* 종류
  * `Extractive Answer Datasets` : 질의(Question)에 대한 답이 항상 주어지는 지문(Context)의 Segment(or span)으로 존재
    * Cloze Tests : CNN/Daily Mail, CBT 등
    * Span Extraction : SQuAD, KorQuAD, NewsQA, Natural Questions 등
  * `Descriptive/Narrative Answer Datasets` : 답이 지문 내에서 추출한 span이 아니라, 질의를 보고 생성된 sentence(or free-form)의 형태
    * MS MARCO, Narrative QA 등
  * `Multi-choice Datasets` : 질의에 대한 답을 여러 개의 answer candidates 중 하나로 고르는 형태
    * MCTest, RACE, ARC 등
* Challenges in MRC
  * 단어들의 구성이 유사하지는 않지만 동일한 의미의 문장을 이해
    * DuoRC(paraphrased paragraph)
    * QuoRef(coreference resolution)
  * Unanswerable questions (Question with 'No Answer') : 주어진 지문에서 질문에 대한 답을 찾을 수 없는 경우
    * SQuAD 2.0
  * Multi-hop reasoning : 여러 개의 document에서 질의에 대한 supporting fact를 찾아야지만 답을 찾을 수 있음
    * HotpotQA, QAngaroo
* MRC의 평가 방법
  * `Exact Match(EM)` & `F1 Score` : Extractive answer datasets, Multi-choice answer datasets
    * Exact Match(EM) or Accuracy : 예측한 답과 ground-truth가 정확히 일치하는 샘플의 비율
    * F1 Score : 예측한 답과 ground-truth 사이의 token overlap을 F1으로 계산
  * `ROUGE-L` & `BLUE` : Descriptive answer datasets
    * ROUGE-L Score : 예측한 값과 ground-truth 사이의 overlap recall - LCS(Longest Common Subsequence) 기반
    * BLUE(Bilingual Evaluation Understudy) : 예측한 답과 ground-truth 사이의 precision - n-gram 기반

### Unicode & Tokenization

* `Unicode` : 전 세계의 모든 문자를 일관되게 표현하고 다룰 수 있도록 만들어진 문자셋, 각 문자마다 숫자 하나에 매핑한다.
  * 인코딩 : 문자를 컴퓨터에서 저장 및 처리할 수 있게 이진수로 바꾸는 것
    * UTF-8 (Unicode Transformation Format) : 현재 가장 많이 쓰이는 인코딩 방식, 문자 타입에 따라 다른 길이의 바이트를 할당한다.
  * Python에서 Unicode 다루기
    * Python3부터 string 타입은 유니코드 표준을 사용한다.
    * `ord('A')` : 문자를 유니코드 code point로 변환한다.
    * `chr(65)` : code point를 문자로 변환한다.
  * 한국어는 한자 다음으로 유니코드에서 많은 코드를 차지함
    * 완성형 : 모든 완성형 한글 11,172자 `len('가') == 1`
    * 조합형 : 조합하여 글자를 만들 수 있는 초/중/종성 `len('가') == 2`
* `Tokenization` : 텍스트를 토큰 단위로 나누는 것 (단어, 형태소, subword 등)
  * Subword tokenization : 자주 쓰이는 글자 조합은 한 단위로 취급하고, 자주 쓰이지 않는 조합은 subword로 쪼갠다.
  * BPE (Byte-Pair Encoding)
    * 데이터 압축용으로 제안된 알고리즘
    * NLP에서 토크나이징용으로 활발하게 사용되고 있다.
    * 가장 자주 나오는 글자 단위 Bigram (or Byte pair)를 다른 글자로 치환하고, 저장해두는 과정을 반복한다.
    * Out-Of-Vocabulary(OOV) 문제를 해결해주고 정보학적으로 이점을 가진다.
    * BPE 방법론 중 하나가 WordPiece Tokenizer

### Looking into the Dataset

* `KorQuAD`
  * LG CNS가 AI 언어지능 연구를 위해 공개한 질의응답/기계독해 한국어 데이터셋
  * 인공지능이 한국어 질문에 대한 답변을 하도록 필요한 학습 데이터셋
* 데이터셋 설명
  * 1,550개의 위키피디아 문서에 대해서 10,649 건의 하위문서들과 크라우드 소싱을 통해 제작한 63,952 개의 질의응답쌍으로 구성되어 있음 (TRAIN 60,407 / DEV 5,774 / TEST 3,898)
  * 데이터 수집 과정 : SQuAD 1.0 데이터 수집 방식을 벤치마크하여 표준성 확보
* 데이터셋 다운 방법
  ```python
  from datasets import load_dataset
  dataset = load_dataset('squad_kor_v1', split='train')
  ```
* 데이터셋 탐색
  * Strong Supervision : `answer_start`가 주어짐 (주어지지 않는 경우 Distant Supervision)
  * 질문 유형 : 구문 변형 / 어휘 변형(유의어, 일반상식) / 여러 문장의 근거 활용 / 논리적 추론 등
  * 답변 유형 : 대상 / 인물 / 시간 / 장소 / 방법 / 원인
* **Further Reading**
  - [문자열 type에 관련된 정리글](https://kunststube.net/encoding/)
  - [KorQuAD 데이터 소개 슬라이드](https://www.slideshare.net/SeungyoungLim/korquad-introduction)
  - [Naver Engineering: KorQuAD 소개 및 MRC 연구 사례 영상](https://tv.naver.com/v/5564630)

## [2강] Extraction-based MRC

### Extraction-based MRC

* 질문(question)의 답변(answer)이 항상 주어진 지문(context)내에 span으로 존재
* → 지문(context) 내 답의 위치를 예측 (분류 문제)
* 평가방법 : EM, F1
* 모델 구조 : PLM + Classifier

### Pre-processing

* Tokenization : WordPiece Tokenizer 사용
* Special Tokens : Question과 Context 구분
  * `[CLS] ~Question 문장~ [SEP] ~Context 내용~`
* Attention Mask : 입력 시퀀스 중에서 attention을 연산할 때 무시할 토큰(`[PAD]` 등)을 0, 1로 표시
* Token Type Ids : 입력이 2개 이상의 시퀀스(질문 & 지문) 일 때, 각각에게 ID를 부여하여 모델이 구분해서 해석하도록 유도
* 출력 표현 - 정답 출력 : 문서 내 존재하는 연속된 단어토큰(span)의 시작과 끝 위치를 예측하도록 학습 → Token Classification 문제로 치환!

### Fine-tuning

* 실제 답의 start/end 위치와 cross-entropy loss를 계산
  ```python
  start_loss = loss_fct(start_logits, start_positions)
  end_loss = loss_fct(end_logits, end_positions)
  total_loss = (start_loss + end_loss) / 2
  ```

### Post-processing

* 불가능한 답 제거하기
  * end가 start보다 앞에 있는 경우
  * context를 벗어나는 경우
  * max_answer_length보다 긴 경우
* 최적의 답안 찾기
  * 예측된 start/end position 에서 score (logits)가 가장 높은 N개를 찾는다.
  * 불가능한 답을 제거한다.
  * 가능한 조합 중 score의 합이 큰 순서대로 정렬하여 최종 예측을 선정한다.
* **Further Reading**
  - [SQuAD 데이터셋 둘러보기](https://rajpurkar.github.io/SQuAD-explorer/)
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning](http://jalammar.github.io/illustrated-bert/))
  - [Huggingface datasets](https://huggingface.co/datasets)

## [3강] Generation-based MRC

### Generation-based MRC

* 질문(question)의 답변(answer)이 주어진 지문(context)내에 존재할 수도 있고 아닐 수도 있는 경우
* → 주어진 지문과 질의(question)를 보고, 답변을 생성 (생성 문제)
* 평가방법 : EM, F1 (Extractive answer datasets → Extraction-based MRC와 동일한 평가 방법 사용)
* 모델 구조 : Seq-to-seq PLM

### Pre-processing

* Tokenization : Extraction-based MRC와 동일하게 WordPiece Tokenizer 사용
* Special Tokens : `[CLS]`, `[SEP]` 토큰 대신 자연어(question, context 단어 그대로)를 이용한다.
* Attention Mask : Extraction-based MRC와 동일하게 사용
* Token Type Ids : 입력 시퀀스에 대한 구분이 없기 때문에 사용하지 않음
* 출력 표현 - 정답 출력 : 전체 시퀀스의 각 위치마다 모델이 아는 모든 단어들 중 하나의 단어를 맞추는 classification 문제

### Model

* BART : 기계 독해, 기계 번역, 요약, 대화 등 Seq-to-seq 문제의 pre-training을 위한 denoisiong autoencoder
  * BERT의 Bidirectional Encoder + GPT의 Autoregressive(Uni-directional) Decoder
  * 텍스트에 노이즈를 주고 원래 텍스트를 복구하는 문제를 푸는 것으로 pre-training 한다.

### Post-processing

* Searching
  * Greedy Search : 현재 가장 높은 가능성을 선택한다.
  * Exhaustive Search : 모든 가능성을 본다.
  * **Beam Search** : Exhaustive Search를 하되, 각 time step마다 top-k 만 남기는 방식
* Further Reading
  * [Introducing BART](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html)
  * [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
  * [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683)

## [4강] Passage Retrieval - Sparse Embedding

### Introduction to Passage Retrieval

* Open-domain Question Answering : 대규모의 문서 중에서 질문에 대한 답을 찾기
  * Stage 1 : Query -> **Retrieval System** -> Retrieved Passage
  * Stage 2 : Query + Retrieved Passage -> **MRC Model** -> Answer
* Passage Retrieval
  * 질문(query)에 맞는 문서(passage)를 찾는 것
  * query와 passage를 임베딩한 뒤 유사도로 랭킹을 매기고, 유사도가 가장 높은 passage를 선택함

### Passage Embedding and Sparse Embedding

* Passage Embedding Space : Passage Embedding의 벡터 공간
* Sparse Embedding
  * Bag-of-Words (BoW) : 특정 단어가 존재하는지 아닌지를 0,1로 표현
    * BoW를 구성하는 방법 : unigram(1-gram), bigram(2-gram)
    * Term Value를 결정하는 방법 : Term이 document에 등장하는지, Term이 몇번 등장하는지 등
  * 특징
    * Dimension of embedding vector = number of terms
      * n-gram에서 n이 늘어날수록 vocab size가 기하급수적으로 커진다.
      * 등장하는 단어가 많아질수록 증가한다.
    * 의미(semantic)가 비슷하지만 다른 단어인 경우 비교가 불가

### TF-IDF

* TF-IDF (Term Frequency-Inverse Document Frequency)
  * Term Frequency (TF) : 단어의 등장빈도 (보통 raw count 후 nomalize해서 사용)
  * Inverse Document Frequency (IDF) : 단어가 제공하는 정보의 양
    * $IDF(t) = \log \frac{N}{DF(t)}$
    * Document Frequency (DF) : Term $t$가 등장한 document의 개수
    * N : 총 document의 개수
  * $TF(t,d) \times IDF(t)$
    * Low TF-IDF : 관사, 조사 등
    * High TF-IDF : 고유 명사 등
* TF-IDF를 이용한 유사도 계산
  * $Score(D, Q) = \sum_{term \in Q} TFIDF(term, Q) * TFIDF(term, D)$

### BM25

* 개념 : TF-IDF 개념 + 문서의 길이까지 고려 + TF 값에 일정한 범위 지정
  * $Score(D,Q) = \sum_{term \in Q} IDF(term) \cdot \frac{TFIDF(term, D) \cdot (k_1+1)}{TFIDF(term,D)+k_1 \cdot (1-b+b \cdot \frac{\| D \|}{avgdl})}$
* 특징
  * 평균적인 문서의 길이보다 작은 문서에서 단어가 매칭된 경우 그 문서에 대해 가중치 부여
  * 실제 검색엔진, 추천 시스템 등에서 많이 사용되고 있다.
* Further Reading
  - [Pyserini BM25 MSmarco documnet retrieval 코드](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-doc.md)
  - [Sklearn feature extractor](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) ⇒ text feature extractor 부분 참고

## [5강] Passage Retrieval - Dense Embedding

### Introduction to Dense Embedding

* Passage Embedding : 구절(passage)을 벡터로 변환하는 것
* Sparse Embedding
  * 장점
    * 중요한 term 들이 정확히 일치해야 하는 경우 성능이 뛰어남
  * 단점
    * 차원의 수가 매우 크다 (-> compressed format으로 극복 가능)
    * (비슷한 의미를 가지는 두 단어 사이의) 유사성을 고려하지 못한다.
    * 임베딩이 구축되고 나서는 학습이 불가능하다.
* Dense Embedding
  * 장점
    * 더 작은 차원의 고밀도 벡터 : 각 차원이 특정 term에 대응되지 않고, 대부분의 요소가 non-zero 값
    * 단어의 유사성 또는 맥락을 파악해야 하는 경우 성능이 뛰어남
    * 학습을 통해 임베딩을 만들며 추가적인 학습 또한 가능

### Training Dense Encoder

* Dense Encoder
  * BERT와 같은 Pre-trained language model (PLM)이 자주 사용, 그 외 nueral network 구조도 가능
  * BERT 사용시 `[CLS]` token의 output 사용!
* Dense Encoder 구조
  * Question q -> Question encoder (BERT_Q) -> $h_q$
  * Passage p -> Passage encoder (BERT_P) -> $h_p$
  * Similarity score (ex. dot product) : $sim(q,p) = h_q^T h_p$
* Dense Encoder 학습 (두 인코더를 Fine-tuning)
  * 목표 : 연관된 question, passage의 dense embedding 거리를 좁히는 것(higher similarity)
  * Negative Sampling
    * 연관되지 않은 question, passage 간의 embedding 거리는 멀어야 한다!
    * 높은 TF-IDF 스코어를 가지지만 답을 포함하지 않는 샘플로 구성하면 좋다.
  * Objective function
    * Positive passagee 에 대한 negative log likelihood (NLL) loss 사용
    * $D = \{ < q_i, p_i^{+}, p_{i,1}^{-}, ... p_{i,n}^{-} > \}_{i=1}^m$
      * $q_i$ : Question
      * $p_i^{+}$ : Positive P
      * $p_{i,n}^{-}$ : Negative P
    * $L(q_i, p_i^{+}, p_{i,1}^{-}, ... p_{i,n}^{-}) = - \log \frac{e^{sim(q_i, p_i^{+})}}{e^{sim(q_i, p_i^{+})} + \sum_{j=1}^{n} e^{sim(q_i, p_{i,j}^{-})}}$
  * Evaluation Metric for Dense Encoder
    * Top-k retrieval accuracy : retrieve 된 passage 중에 답을 포함하는 passage의 비율

### Passage Retrieval with Dense Encoder

* Inference
  1. passage와 query를 각각 embedding 한 후, query로부터 가까운 순서대로 passage의 순위를 매긴다.
  2. retriever를 통해 찾아낸 passage를 활용하여, MRC 모델로 답을 찾는다.
* Dense Encoding 개선
  * 학습 방법 개선 : DPR
  * 인코더 모델 개선 : BERT 보다 크고 정확한 Pretrained 모델
  * 데이터 개선 : 더 많은 데이터, 전처리 등
* Further Reading
  - [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
  - [Open domain QA tutorial: Dense retrieval](https://github.com/danqi/acl2020-openqa-tutorial/blob/master/slides/part5-dense-retriever-e2e-training.pdf)

## [6강] Scaling up with FAISS

### Passage Retrieval and Similarity Search

* MIPS(Maximum Inner Product Search)
  * 주어진 질문(query) 벡터 q에 대해 Passage 벡터 v들  중 가장 질문과 관련된 벡터를 찾아야 한다. → 내적(inner product)이 가장 큰 것을 고른다.
    * $\underset{v_i \in V}{argmax} (q^T v_i)$
* 인덱싱(Indexing) : 방대한 양의 passage 벡터들을 저장하는 방법
* 검색(search) : 인덱싱된 벡터들 중 질문 벡터와 가장 내적값이 큰 상위 k개의 벡터를 찾는 과정
  * brute-force(exhaustive) search : 저장해둔 모든 Sparse/Dense 임베딩에 대해 일일히 내적값을 계산하여 가장 값이 큰 passage 추출 → 문서 양이 방대해지면 오래 걸려서 비효율적임
* Tradeoffs of similarity search
  * Search Speed : 가지고 있는 벡터량이 클 수록 더 오래 걸린다. => **Pruning**
  * Memory Usage : 벡터를 디스크에서 계속 불러오면 속도가 느려지고, RAM에 올리려면 많은 용량이 필요하다. => **Compression**
  * Accuracy : 속도를 증가시키려면 정확도를 희생해야 한다. => **Exhaustive search**

### Approximating Similarity Search

* Compression : vector를 압축하여, 하나의 vector가 적은 용량을 차지
  * Scalar Quantization (SQ) : **4-byte** floating point => **1-byte(8bit)** unsigned integer 로 압축! (SQ8)
* Pruning : Search space를 줄여 search 속도 개선 (dataset의 subset만 방문)
  * Clustering : 전체 vector space를 k개의 cluster로 나눔 (k-means clustering)
  * Inverted File (IVF) : 각 cluster의 centroid id와 해당 cluster의 vector들이 연결되어있는 형태

### Introduction to FAISS

* FAISS : Fast Approximation을 위한 라이브러리, Large Scale에 특화되어 있음
  * 참고링크 : [[FAISS blog](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) \| [FAISS github](https://github.com/facebookresearch/faiss) \| [FAISS tutorial](https://github.com/facebookresearch/faiss/tree/master/tutorial/python) \| [Getting started with Faiss](https://www.pinecone.io/learn/faiss-tutorial/)]

* Passage Retrieval with FAISS
  1. Train index and map vectors : Clustering + IVF + SQ8
     ```python
     index.train() # Train index : Add 할 index 중 일부나 전체로 학습
     index.add() # Add index
     ```
  2. Search based on FAISS index
     ```python
     index.nprobe = 10 # 몇 개의 가장 가까운 cluster를 방문하여 search할 것인지
     index.search()
     ```

### Scaling up with FAISS

* FAISS Basic
  ```python
  index = faiss.IndexFlatL2(d) # 인덱스 빌드하기
  index.add(xb) # 인덱스에 벡터 추가하기
  D, I = index.search(xq, k) # 검색하기
  ```
* IVF with FAISS
  ```python
  quantizer = faiss.IndexFlatL2(d)
  index = faiss.IndexIVFFlat(quantizer, d, nlist) # Inverted File 만들기
  index.train(xb) # 클러스터 학습하기
  index.add(xb) # 클러스터에 벡터 추가하기
  D, I = index.search(xq, k) # 검색하기
  ```
* IVF-PQ with FAISS : 벡터 압축 기법(PQ)를 활용하여 압축된 벡터만 저장
  ```python
  quantizer = faiss.IndexFlatL2(d)
  index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) # 각각의 sub-vector가 8bit로 인코딩됨
  index.train(xb) # 클러스터 학습하기
  index.add(xb) # 클러스터에 벡터 추가하기
  D, I = index.search(xq, k) # 검색하기
  ```
* Using GPU with FAISS : GPU의 빠른 연산 속도를 활용할 수 있으나, 메모리 제한이나 random access 시간이 느리다는 단점 존재
  ```python
  res = faiss.StandardGpuResources() # 단일 GPU 사용하기
  index_flat = faiss.IndexFlatL2(d) # 인덱스 (CPU) 빌드하기
  gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat) # GPU 인덱스로 옮기기
  gpu_index_flat.add(xb)
  D, I = gpu_index_flat.search(xq, k) # 검색하기
  ```
* Using Multiple GPUs with FAISS
  ```python
  cpu_index = faiss.IndexFlatL2(d)
  gpu_index = faiss.index_cpu_to_all_gpus(cpu_index) # GPU 인덱스로 옮기기
  gpu_index.add(xb)
  D, I = gpu_index.search(xq, k) # 검색하기
  ```

## [7강] Linking MRC and Retrieval

### Introduction to Open-domain Question Answering (ODQA)

* MRC : 지문이 주어진 상황에서 질의응답
* ODQA : 지문이 따로 주어지지 않은 상황에서 질의응답 (Linking MRC and Retrieval)
  * Modern search engines : 연관문서 뿐만 아니라 질문의 답을 같이 제공
* History of ODQA
  * Text retrieval conference (TREC) - QA Tracks (1999~2007)
    * 연관문서만 반환하는 information retrieval(IR)에서 더 나아가서, short answer with support 형태가 목표
    1. Question processing : 질문으로부터 키워드를 선택 / Answer type selection 등 룰베이스 방식
    2. Passage retrieval : 기존의 IR 방법을 활용해서 연관된 document를 뽑고, passage 단위로 자른 후 선별 (hand-crafted features 활용)
    3. Answer processing : hand-crafted features와 heuristic을 활용한 classifier를 이용하여, 주어진 question과 선별된 passages 내에서 답을 선택
  * IBM Watson (2011)
    * The DeepQA Project, Jeopardy! (TV quiz  show) 우승
    * 아직 딥러닝보다는 초기 머신러닝 모델들 활용

### Retriever-Reader Approach

* Retriever : 데이터베이스에서 관련있는 문서를 검색(search) 함
* Reader : 검색된 문서에서 질문에 해당하는 답을 찾아냄
* Distant supervision
  * 질문-답변만 있는 데이터셋(CuratedTREC, WebQuestions, WikiMovies)에서 MRC 학습 데이터 만들기 (학습 데이터를 추가하기 위해 활용)
  * 위키피디아에서 Retriever을 이용해 관련성 높은 문서 검색 후 부적합한 문서 제외하고 질문과 연관성이 높은 단락을 supporting evidence로 사용함
* Inference
  * Retriever가 질문과 가장 관련성 높은 문서 n개 출력
  * Reader는 n개 문서를 읽고 답변 예측
  * Reader가 예측한 답변 중 가장 score가 높은 것을 최종 답으로 선택

### Issues and Recent Approches

* Different granularities of text at indexing time
  * 위키피디아에서 각 Passage의 단위를 문서(Article), 단락(Paragraph), 또는 문장(Sentence)으로 정의할지 정해야 함
  * Retriever 단계에서 몇 개(top-k)의 문서를 넘길지 정해야 함 (ex. artical->5, paragraph->29, sentence->78)
* Single-passage training vs Multi-passage training
  * Single-passage : 하나의 passage에서 answer span을 만들고, 여러 passage들이 만든 것 중에서 높은 점수를 가진 answer span 을 고른다.
  * Multi-passage : 여러 passage들을 하나로 취급하고 answer span을 찾도록 한다. (더 많은 메모리, 연산량 소요)
* Importance of each passage
  * Retriever 모델에서 추출된 top-k passage들의 retrieval score를 reader 모델에 전달
* Further Reading
  - [Reading Wikipedia to Answer Open-domain Questions](https://arxiv.org/abs/1704.00051)
  - [A survey on Machine Reading Comprehension](https://arxiv.org/abs/2006.11880)
  - [ACL 2020 ODQA tutorial](https://slideslive.com/38931668/t8-opendomain-question-answering)

## [8강] Reducing Training Bias

### Definition of Bias

* Bias in learning
  * inductive bias : 학습할 때 과적합을 막거나 사전 지식을 주입하기 위해 특정 형태의 함수를 선화는 것
* A Biased World
  * historical bias : 현실 세계가 이미 편향
  * co-occuuence bias : 성별과 직업 등 표면적 상관관계
* Bias in Data Generation
  * specification bias : 입력과 출력 정의 방식으로 인한 편향
  * sampling bias : 데이터 샘플링 방식으로 인한 편향
  * annotator bias : 어노테이션 특성으로 인한 편향

### Bias in Open-domain Qustion Answering

* Training bias in reader model
  * 학습 때 항상 정답이 문서 내에 포함된 데이터쌍만 보게 된다.
  * 학습과 추론 시에 전혀 다른 주제의 문서가 주어진다면 새로운 문서 독해 능력이 매우 떨어질 것이다.
* How to mitigate training bias
  * Train negative examples
    * 훈련할 때 잘못된 예시를 보여줘야 retriever이 negative한 내용들을 먼 곳에 배치할 수 있을 것이다.
    * 이때 negative sample 사이의 차이 정도도 고려해야 한다. (헷갈리는 negative 샘플 뽑기)
      * 높은 BM25 / TF-IDF 매칭 스코어를 가지지만, 답을 포함하지 않는 샘플
      * 같은 문서에서 나온 다른 Passage/Question 선택하기
  * Add no answer bias
    * 답변이 문서 내에 없는 경우에 no answer를 줄 수 있어야 한다.

### Annotation Bias from Datasets

* Annotation bias : 데이터 제작(annotation) 단계에서 bias가 발생하는 것
  * ODQA 학습 시 기존의 MRC 데이터셋 활용
    * Information-seeking questions : 질문하는 사람이 답을 알고 있지 않음
      * Tool을 이용해서 질문의 답을 찾음 : Natural Questions / WebQustions / CuratedTrec => 어떠한 tool을 사용하느냐에 따라 bias가 발생할 수 있다.
    * 질문하는 사람이 답을 알고 있는 상태로 질문 : TriviaQA / SQuAD => 질문과 evidence 문단 사이의 많은 단어가 겹치는 bias 발생 가능
    ![image](https://user-images.githubusercontent.com/35680202/137672776-6155895b-7ba7-4627-9ee7-9055c48e4540.png)
  * Dealing with annotation bias
    * Supporting evidence가 주어지지 않은, 실제 유저의 question을 모아서 데이터셋 구성 [[참고](https://ai.google.com/research/NaturalQuestions/visualization)]
* Another bias in MRC dataset
  * SQuAD : Passage가 주어지고, 주어진 passage 내에서 질문과 답을 생성 => ODQA에 applicable 하지 않은 질문들이 존재
* Further Reading
  - [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300)
  - [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

## [9강] Closed-book QA with T5

### Closed-book Question Answering

* Idea of Closed-book Qustion Answering : 대량의 지식을 사전학습한 언어모델 자체가 이미 하나의 knowledge storage이기 때문에 다른 곳에서 지식을 가져올 필요가 없을 것이다.

### Text-to-Text Format

* Closed-book QA as Text-to-Text Format
  * Closed-book QA에 사용되는 방법은 Generation-based MRC와 유사함
  * 단, 입력에 지문(Context)가 없이 질문만 들어간다는 것이 차이점
  * 사전학습된 언어모델은 BART와 같은 seq-to-seq 형태의 Transformer 모델 사용
  * 각 입력값(질문)과 출력값(답변)에 대한 설명을 맨 앞에 추가하기
* T5 : Text-to-Text format 이라는 형태로 데이터의 입출력을 만들어 거의 모든 자연어처리 문제를 해결하도록 학습된 seq-to-seq 형태의 Transformer 모델
  * 모델이 커서 웬만한 gpu에서는 T5-base 사용

### Experiment Results & Analysis

* Dataset : Open-domain QA 데이터셋 또는 MRC 데이터셋에서 지문을 제거하고 질문과 답변만 남긴 데이터셋 활용
* **Salient Span Masking** : 고유명사, 날짜 등 의미를 갖는 단위에 속하는 토큰 범위를 마스킹한 뒤 학습 (pre-trained 모델에서 추가로 pre-training 진행) => 성능을 크게 끌어올림!
* Fine-tuning : Open-domain QA 데이터셋으로 추가학습
* [[예시](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-trivia.ipynb)]

* Further Reading
  - [Exploring the limits of transfer learning with a unified text-to-text transformer(T5)](https://arxiv.org/abs/1910.10683)
  - [How much knowledge can you pack into the parameters of language model?](https://arxiv.org/abs/2002.08910)
  - [UNIFIEDQA: Crossing Format Boundaries with a Single QA System](https://arxiv.org/abs/2005.00700)

## [10강] QA with Phrase Retrieval

* **Further Reading**
  - [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index](https://arxiv.org/abs/1906.05807)
  - [Contextualized Sparse Representations for Real-Time Open-Domain Question Answering](https://arxiv.org/abs/1911.02896)
