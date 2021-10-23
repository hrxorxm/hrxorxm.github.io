---
layout: post
title:  "[Boostcamp AI Tech] 6주차 - NLP"
subtitle:   "Naver Boostcamp AI Tech Level2 U Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level2-U-Stage]
use_math: true
---

# 부스트캠프 6주차

## [1강] Introduction to NLP & Bag-of-Words

### NLP 분야

* NLP(Natural language processing) (major conferences: ACL, EMNLP, NAACL)
  * 종류
    * NLU(Natural Language Understanding) : 컴퓨터가 자연어를 이해하는 것
    * NLG(Natural Language Generation) : 자연어를 상황에 따라 적절히 생성
  * Low-level parsing
    * Tokenization : 주어진 문장을 단어 단위로 쪼개나가는 과정
    * stemming : 단어의 어근을 추출하는 과정
  * Word and phrase level
    * NER(Named entity recognition) : 고유명사 인식
    * POS(part of speech) tagging : 문장 내 품사나 성분을 알아내는 것
    * noun-phrase chunking, dependency parsing, coreference resolution 등
  * Sentence lebel
    * Sentiment analysis : 긍정/부정 감정분석
    * machine translation : 기계번역
  * Multi-sentence and paragraph level
    * Entailment prediction : 두 문장간의 논리적인 내포, 모순관계 파악
    * question answering : 독해기반의 질의응답
    * dialog systems : 챗봇과 같이 대화를 수행할 수 있는 기술
    * summarization : 주어진 문서를 요약
* Text mining (major conferences: KDD, The WebConf(formerly, WWW), WSDM, CIKM, ICWSM)
  * 텍스트, 문서 데이터를 통해 유용한 (주로 사회과학적인) 인사이트를 얻는 것
  * Document clustering (topic modeling) : 문서 군집화
* Information retrieval (major conferences: SIGIR, WSDM, CIKM, RecSys)
  * 구글, 네이버 등이 사용하는 검색기술
  * 추천 시스템 기술 : 사용자의 수동 검색 없이 정보 제공, 적극적이고 자동화된 검색 시스템의 새로운 형태

### NLP 발전 과정

* 주어진 텍스트를 벡터로 나타내는 기술 : Word2Vec / GloVe
* 주요 아키텍쳐 : RNN-family models : LSTMs, GRUs
* 트랜스포머의 등장 : attention modules, Transformer models
* self-supervised 방식으로 사전학습 시킨 범용적인 모델 : BERT, GPT-3
* 위 사전학습 모델을 이용한 전이학습

### 문서 분류 예제

* Bag-of-Words
  * Step 1 : unique한 단어들을 모아서 vocabulary를 만든다.
  * Step 2 : 각 categorical 변수들을 one-hot 벡터로 표현한다.
    * 단어의 의미에 상관없이 모두 동일한 관계를 가진 형태가 된다.
    * 단어 간의 거리는 모두 $\sqrt{2}$, 단어 간의 코사인 유사도는 모두 0
  * Step 3 : 문장/문서에 나온 단어들의 one-hot 벡터들을 모두 더한다. = Bag-of-Words 벡터
* NaiveBayes Classifier for Document Classification
  * 문서 d, 클래스 c
  * $C_{MAP} = \underset{c \in C}{argmax} P(c \| d)$ ← MAP(Maximum a posterion) : 가장 예상되는 클래스
    * $= \underset{c \in C}{argmax} \frac{P(d \| c) P(c)}{P(d)}$ ← Bayes Rule
    * $= \underset{c \in C}{argmax} P(d \| c) P(c)$ ← Dropping the denominator
  * $P(d \| c) p(c) = P(w_1, w_2, ..., w_n) P(c) \rightarrow P(c) \Pi_{w_i \in W} P(w_i \| c)$ ← 조건부 독립 가정
  * 특정한 단어가 전혀 나타나지 않았을 경우 그 클래스가 될 확률이 무조건 0이 되는 경우가 발생할 수 있으므로 추가적인 regularizer와 함께 쓰이기도 한다.

## [02강] Word Embedding

* Word Embedding : 각 단어들을 특정 차원으로 이루어진 공간상의 한 점의 좌표를 나타내는 벡터로 변환해주는 기법

### [Word2Vec](https://arxiv.org/abs/1310.4546)

* 알고리즘
  * 비슷한 의미를 가진 단어가 좌표공간 상에서 비슷한 위치로 맵핑되도록 하기 위한 알고리즘
  * 같은 문장에서 나타난 인접한 문장들 간의 의미가 비슷할 것이라는 가정
  * 한 단어의 주변에 등장하는 단어들을 통해 그 단어의 의미를 알 수 있다.
* 학습 방법
  * 단어 주변에 나타나는 단어들의 확률 분포를 예측한다.
  * 단어를 입력으로 주고 주변 단어들을 가린 채 이를 예측하도록 하는 방식으로 학습

### [GloVe](https://aclanthology.org/D14-1162/)

* Global Vectors for Word Representation
* 새로운 형태의 loss function 사용, 단어가 같이 등장한 횟수를 미리 계산 (학습이 더 빠름)
* $J(\Theta) = \frac{1}{2} \sum_{i,j=1}^{W} f(P_{ij})(u_i^T v_j - \log{P_{ij}})^2$

## [3강] Recurrent Neural Network and Language Modeling

* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [CS231n(2017) Lecture10 RNN](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)

### Vanilla RNN

* $h_t = f_W (h_{t-1}, x_t) = \tanh (W_{hh} h_{t-1} + W_{xh} x_t)$
* $y_t = W_{hy} h_t$

### Types of RNNs

* one-to-one : Standard neural networks
* one-to-many : Image Captioning
* many-to-one : Sentiment Classification
* many-to-many
  * input sequence를 다 읽고 예측 : Machine Translation
  * 각 time step마다 예측 : POS tagging, Video classification on frame level

### Character-level Language Model

* "hello" -> vocab=[h, e, l, o] -> ont-hot vector로 변환 -> training
* $h_t = f_W (h_{t-1}, x_t) = \tanh (W_{hh} h_{t-1} + W_{xh} x_t + b)$
* Logit $= W_{hy} h_t + b$ -> softmax 후 target char 예측
* Backpropagation through time (BPTT)
* Searching for Interpretable Cells : 필요한 정보가 어디에 저장되어있는지 파악하기 위해서, hidden state의 차원 하나를 고정하고, 그 값이 time step이 진행됨에 따라 어떻게 변하는지 확인 하기
* Vanishing/Exploding Gradient Problem in RNN

## [4강] LSTM and GRU

* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Long Short-Term Memory (LSTM)

* 핵심 아이디어
  * long-term dependency 문제를 해결하기 위함 (단기 기억을 길게 기억할 수 있도록 개선)
  * cell state information을 변형없이 pass할 수 있도록 함
*  $\{ C_t, h_t \} = LSTM(x_t, C_{t-1}, h_{t-1})$
* $\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{pmatrix} W \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix}$
  * input gate : $f_i = \sigma (W_i \cdot [h_{t-1}, x_t] + b_i)$
  * forget gate : $f_t = \sigma (W_f \cdot [h_{t-1}, x_t] + b_f)$
  * output gate : $f_o = \sigma (W_o \cdot [h_{t-1}, x_t] + b_o)$
  * gate gate : $\tilde{C_t} = \tanh (W_C \cdot [h_{t-1}, x_t] + b_C)$
* $C_t = f \odot C_{t-1} + i \odot g = f_t \cdot C_{t-1} + i_t \cdot \tilde{C_t}$
* $h_t = o_t \odot \tanh(C_t)$

### Gated Recurrent Unit (GRU)

* 핵심 아이디어
  * LSTM의 모델 구조를 경량화
  * 동작원리는 비슷하지만 오직 hidden state만 존재
* $z_t = \sigma (W_z \cdot [h_{t-1}, x_t])$
* $r_t = \sigma (W_r \cdot [h_{t-1}, x_t])$
* $\tilde{h_t} = \tanh(W \cdot [r_t \cdot h_{t-1}, x_t])$
* $h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h_t}$
* Backpropagation in LSTM & GRU : 덧셈 연산으로 인해 gradient 복사 효과로 인해서 gradient를 큰 변형없이 멀리 있는 time step까지 전달해줄 수 있다.

## [5강] Sequence to Sequence with Attention

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
* [CS224n(2019) Lecture8 NMT](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)

### Seq2Seq Model

* Sequence-to-Sequence : RNN의 구조 중에서 many to many의 형태 (입력 시퀀스를 모두 읽은 후 출력 시퀀스를 생성/예측하는 모델)
* ![image](https://user-images.githubusercontent.com/35680202/132442735-e004a799-a2bf-423a-a571-4f80933a873c.png)
  * Encoder / Decoder : 서로 share 하지 않는 파라미터를 갖는 RNN 모듈 (LSTM 모듈)
  * `<start>` `<end>` 토큰
* 문제 : Encoder의 RNN 모듈의 마지막 hidden state에만 의존한다면, 입력 문장의 길이가 길어질 때, dimension이 한정되어 있어서 정보가 다 담기지 못할 수 있다.

### Seq2Seq Model with Attention

* 해결 : 각 time step에서 나온 hidden state들을 이용하자 (=>Attention)
* 방법 : Decoder에서 각 단어를 생성할 때 필요로하는 Encoder의 hidden state vector를 적절히 선택해서 예측에 활용한다.
* ![image](https://user-images.githubusercontent.com/35680202/132442525-cf5b6bc5-1d16-4325-b94c-c7f47f54059a.png)
  * Attention scores : Decoder의 hidden state vector와 Encoder의 hidden state vector들을 내적한 것. (내적 말고 다른 방법이 있을 수 있음 => Attention variants)
  * Attention distridution (=Attention vector) : Attention scores를 확률값(합이 1인 형태)으로 변환시킨 것. Attention distribution는 Encoder hidden state vector에 부여되는 가중치로서 사용된다.
  * Attention output (=Context vector) : Encoder hidden state vector들의 가중 평균을 구한 벡터.
  * Decoder의 hidden state vector와 context vector가 concat 되어서 output layer의 입력으로 들어가서 다음 단어를 예측한다.
  * Teacher forcing : 학습할 때 Decoder에서 각 time step에서 이 전 time step의 예측값이 아니라, 올바른 token을 입력으로 주는 것 (학습이 용이하지만, 테스트 때 괴리가 있을 수 있다.)
* Attention variants (compute $e \in R^{N}$ from $h_1, ..., h_N \in R^{d_1}$ and $s \in R^{d_2}$)
  * (dot) Basic dot-product attention : $e_i = s^T h_i$
    * $d_1 = d_2$라고 가정
  * (general) Multiplicative attention : $e_i = s^T W h_i$
    * $W \in R^{d_2 \times d_1}$ : weight matrix
  * (concat) Addictive attention (=Bahdanau attention) : $e_i = v^T \tanh (W_1 h_i + W_2 s) = v^T \tanh (W [s; h_i])$
* Attention is Great!
  * NMT 성능을 높여주었고, 해석 가능성(interpretability)을 제공한다.
    * [How to Visualize Your Recurrent Neural Network with Attention in Keras](https://medium.com/datalogue/attention-in-keras-1892773a4f22)
    * [Visualizing Attention in PyTorch](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#visualizing-attention)
  * bottleneck problem, vanishing gradient problem 해결

## [6강] Beam search and BLEU score

- [Deep learning.ai-BeamSearch](https://www.youtube.com/watch?v=RLWuzLLSIgw&feature=youtu.be)
- [Deep learning.ai-RefiningBeamSearch](https://www.youtube.com/watch?v=gb__z7LlN_4&feature=youtu.be)
- [OpenNMT-beam search](https://opennmt.net/OpenNMT/translation/beam_search/)

### Beam search

* Greedy decoding
  * 방법 : 현재 time step에서 가장 좋아보이는 단어를 그때그때 선택하는 방식
  * 문제 : 중간에 잘못선택했다는 것을 깨달아도 뒤로 돌아갈 수 없음
* Exhaustive search
  * 방법 : 모든 가능한 시퀀스 y를 계산한다.
    * $P(y \| x) = P(y_1 \| x) P(y_2 \| y1, x) P(y_3 \| y_2, y_1, x) ... P(y_T \| y_1, ..., y_{T-1}, x) = \Pi_{1}^{T} P(y_t \| y_1, ..., y_{t-1}, x)$
    * 위 식을 최대화하는 y를 찾는 것
  * 문제 : 현실적으로 계산량이 너무 많아 불가능
* Beam search
  * 방법 : 두 아이디어의 차선책, top k 개의 가능한 경우의 수를 고려하는 방식
    * k : beam size (일반적으로 5~10)
    * $score(y_1, ..., y_t) = \log P_{LM} (y_1, ..., y_t \| x) = \sum_{i=1}^{t} \log P_{LM} (y_i \| y_1, ..., y_{i-1}, x)$
  * ![image](https://user-images.githubusercontent.com/35680202/132449278-05f6176e-9e5d-46d8-bce9-ff481899e399.png)
  * Stopping criterion
    * 가능한 경우의 수들 중에서 `<end>` 토큰을 다른 시점에 만들어낼 수도 있다. `<end>` 토큰을 만들면 그 경우는 search를 중단(완료)하고 임시 메모리에 저장해둔다.
    * timestep T에 이르렀을 때 멈추거나, n개의 hypotheses가 완료되면 멈춘다.
  * Finishing up
    * 완료된 hypotheses들의 리스트가 있을 때, 가장 높은 점수를 내는 하나를 뽑는다.
    * 음수가 더해지기 때문에 hypotheses가 길수록 score(joint probability)의 값이 작아지고, 이를 해결하기 위해 length 를 이용해서 normalize 한다.
    * $score(y_1, ..., y_t) = \frac{1}{t} \sum_{i=1}^{t} \log P_{LM} (y_i \| y_1, ..., y_{i-1}, x)$

### [BLEU score](https://aclanthology.org/P02-1040.pdf)

* 자연어 생성 모델의 품질, 결과의 정확도를 측정하는 방법
* Precision and Recall
  * 정밀도(Precision) = number_of_correct_words / length_of_prediction
  * 재현율(Recall) = number_of_correct_words / length_of_reference
  * F-measure = 2 x precision x recall / (precision + recall)
    * 산술평균 = $\frac{a + b}{2}$
    * 기하평균 = $(a \times b)^{\frac{1}{2}}$
    * 조화평균 = $\frac{1}{\frac{\frac{1}{a} + \frac{1}{b}}{2}}$
    * 산술평균 >= 기하평균 >= **조화평균**
  * 문제 : 문법적으로 말이 되지 않는 문장을 알 수 없음
* BLEU(BiLingual Evaluation Understudy)
  * N-gram(N개의 연속된 단어)가 겹치는지 평가도 추가 (n = 1~4)
  * precision만 고려하고, recall은 고려하지 않는다.
  * 조화평균이 아닌 **기하평균**을 사용한다.
  * $BLEU = \min (1, \frac{length\_of\_prediction}{length\_of\_reference}) (\Pi_{i=1}^{4} {precision}_i)^{\frac{1}{4}}$

## [7강] Transformer (1)

* RNN Family
  * RNN : Long-term dependency 문제
  * Bi-Directional RNNs : Forward RNN + Backward RNN -> concat

* Transformer
  * 목표
    * 더 이상 RNN, CNN 모듈을 사용하지 않고, Attention만을 이용한 모델 구조
  * 구조
    * Query와 Key는 같은 차원 수 $d_k$ 를 가지고, Value는 $d_v$ 차원이다.
    * Dot-Product Attention
      * 하나의 쿼리 $q$에 대해서 : $A(q, K, V) = \sum_i \frac{\exp(q \cdot k_i)}{\sum_j \exp(q \cdot k_j)} v_i$
      * 여러 쿼리 matrix $Q$에 대해서 : $A(Q, K, V) = softmax(QK^T) V$
        * 결과 차원 수 : $(\| Q \| \times d_k) \times (d_k \times \| K \|) \times (\| V \| \times d_v) = (\| Q \| \times d_v)$
        * $\| Q \|$와 $\| K \|$ 가 같을 필요는 없다. ($\| Q \|$는 그냥 쿼리 수일 뿐)
        * $\| K \|$ (Key의 개수)와 $\| V \|$ (Value의 개수)는 같아야 한다. 
        * Row-wise sorftmax : $QK^T$ 를 계산한 결과에서 row-wise하게 softmax를 취한다.
    * Scaled Dot-Product Attention
      ![image](https://user-images.githubusercontent.com/35680202/133083921-f3f4e76e-69f9-4506-b780-38d765d151ec.png)
      * $A(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$
      * $\frac{1}{\sqrt{d_k}}$ : softmax 값이 쏠리지 않도록 함
* 참고 자료
  * [Attention is all you need](https://arxiv.org/abs/1706.03762)
  * [Illustrated Transformer](https://nlpinkorean.github.io/illustrated-transformer/)
  * [부스트캠프 2주차 Day9 내용 정리](https://hrxorxm.github.io/boostcamp-ai-tech/2021/08/12/Level1-Day9.html#h-8%EA%B0%95-sequential-models---transformer)

## [8강] Transformer (2)

* Multi-Head Attention
  * 목표
    * 동일한 sequence에 대해서 여러 측면에서 병렬적으로 정보를 뽑아오자!
    * 각 헤드가 서로 다른 정보들을 상호보완적으로 뽑아오는 역할
  * 구조
    ![image](https://user-images.githubusercontent.com/35680202/133083669-8e13a3eb-2761-4625-85de-7db25afa8fe0.png)
    * $MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^o$
      * where $head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)$

* 연산량 비교
  ![image](https://user-images.githubusercontent.com/35680202/133038565-17f04f54-c7a4-4e46-9d58-c42122016ca0.png)
  * 수식 설명
    * $n$ : sequence length / $d$ : dimension of representation
    * Self-Attention : $QK^T$에서, $(n \times d) \times (d \times n)$이므로 $O(n^2 \cdot d)$ 이다.
    * Recurrent : $W_{hh} \cdot h_{t-1}$ 에서 $(d \times d) \times (d \times 1)$ 이고, 매 time step에서 계산해줘야하기 때문에, $O(n \cdot d^2)$ 이다.
  * 특징
    * Self-Attention에서 입력 길이가 길어지면 메모리 사용량이 커진다. 하지만 (Sequential Operations 부분을 보면) 병렬화가 가능하기 때문에 학습은 더 빨리 진행될 수 있다.
    * Maximum Path Length : Long-term dependency와 직접적으로 관련, 가장 끝에 있는 단어가 가장 앞에 있는 단어의 정보를 참조하려면 RNN에서는 $n$번의 계산이 필요하지만, Self-Attention에서는 끝에 있는 단어라도 인접한 단어와 차이없이 필요한만큼의 정보를 바로 가져올 수 있다.

* Add (Residual Connection) & Normalize (Layer Normalization)
  ![image](https://user-images.githubusercontent.com/35680202/133040855-0f5e8447-3873-4dca-8f72-c0df35e29379.png)
  * Residual Connection : Gradient Vanishing 문제 해결, 학습 안정화 효과! 입력 벡터와 Self-Attention의 출력 벡터의 dimension이 동일하도록 유지해야한다.
  * Layer Normalization
    ![image](https://user-images.githubusercontent.com/35680202/133082991-c66ed00a-785b-415d-a3f3-bc675c4ea4f6.png)
    * $\mu^l = \frac{1}{H} \sum_{i=1}^{H} a_i^l$, $\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_i^l - \mu^l)^2}$, $h_i = f(\frac{g_i}{\sigma_i} (a_i - \mu_i) + b_i)$
    * each word vectors(특정 단어를 표현하는 노드들)의 평균과 분산을 0, 1로 만든다.
    * 레이어의 각 노드별로 affine transform($y = ax + b$)을 적용시킨다.

* Positional Encoding
  * 순서를 특정지을 수 있는 상수 벡터를 입력 벡터에 더해준다.
  * sin, cos 등으로 이루어진 주기 함수를 이용한다.
  * $PE(pos, 2i) = \sin (pos / 10000^{2i / d_{model}})$
  * $PE(pos, 2i + 1) = \cos (pos / 10000^{2i / d_{model}})$

* Warm-up Learning Rate Scheduler
  ![image](https://user-images.githubusercontent.com/35680202/133088905-b876cdbd-9915-4ef0-a8ac-01201cf60b28.png)
  * $learning\_rate = d_{model}^{-0.5} \cdot \min (step^{-0.5} \cdot warmup\_steps^{-1.5})$

* Encoder Self-Attention Visualization : [예제](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)

* Decoder Masked Self-Attention
  * 예측과정을 생각해보면, 뒤에 나오는 단어는 모르는 상태로 예측해야한다.
  * 따라서 softmax 에 넣기 전에, 뒤에 나오는 단어에 대한 score값들을 `-inf`으로 처리하여, softmax 함수 후 값이 0이 될 수 있도록 한다.

* 추가 자료
  * [Attention is not Explanation](https://arxiv.org/pdf/1902.10186.pdf)
  * [Attention is not not Explanation](https://aclanthology.org/D19-1002.pdf)

## [9강] Self-supervised Pre-training Models

### [GPT-1](https://openai.com/blog/language-unsupervised/)

![image](https://user-images.githubusercontent.com/35680202/133105769-53936ee6-02ac-4c86-8c60-654759fcf37e.png)

* 특징
  * 다양한 special tokens(`<S>, <E>, $`)를 제안함 (다양한 task를 처리할 수 있는 모델을 위해)
  * Attention block을 총 12개 쌓음
* Text Prediction
  * Pretraining task
  * 첫 단어부터 그 다음 단어를 순차적으로 예측하는 태스크
  * => masked self-attention 을 사용함!
* Task Classifier
  * Downstream task : Classification / Entailment / Similarity / Multiple Choice
  * Task Classifier를 새 레이어로 갈아끼우고 Transformer와 함께 다시 학습시킨다. (학습된 Transformer 부분은 learning rate 작게 주기)
  * `Extract` 토큰 : sequence 끝을 나타내는 것 뿐만 아니라, 주어진 여러 입력 문장들로부터 태스크에 필요로 하는 여러 정보들을 추출하는 역할을 한다.
* 한계 : only use left context or right context

### [ELMo](https://arxiv.org/abs/1802.05365)

* Transformer 대신 Bi-LSTM 사용

### [BERT](https://arxiv.org/abs/1810.04805)

![image](https://user-images.githubusercontent.com/35680202/133125898-e00f260e-65c8-46ae-ba3c-1eaf4071d96e.png)

* Pre-training tasks in BERT
  * Masked Language Model(MLM)
    * 입력 토큰들 중에서 k %를 마스킹하여 이 마스킹된 토큰들을 예측하도록 학습 (use k=15%)
    * k가 너무 크면 문맥을 파악하기 위한 정보가 부족해지고, k가 너무 작으면 학습 효율이 너무 떨어진다.
    * downstream task를 수행할 때는 `[MASK]`라는 토큰이 등장하지 않기 때문에,  80%만 `[MASK]` 토큰으로 치환하고, 10%는 랜덤 토큰, 10%는 원래 토큰을 사용한다.
  * Next Sentence Prediction(NSP)
    * 문장 level에서의 task에 대응하기 위한 기법
    * `[SEP]` : 문장을 구분하는, 문장의 끝을 알리는 토큰
    * `[CLS]` : 문장 또는 여러 문장에서의 예측 태스트를 수행하는 역할을 담당하는 토큰 (GPT-1의 Extract 토큰 역할)

* BERT Summary
  * L : Self-Attention block 수 / H : 인코딩 벡터의 차원 수 / A : Attention Head 수
  * BERT BASE : L=12, H=768, A=12
  * BERT LARGE : L=24, H=1024, A=16

* Input Representation
  ![image](https://user-images.githubusercontent.com/35680202/133120845-813d104c-f0a8-436c-b0ad-81e0e292b065.png)
  * (Subword 단위) WordPiece embeddings
  * positional **embedding** vector 도 학습과정을 통해 구한다! (positional **encoding** (X))
  * Segment embedding : 문장 구분

* BERT 와 GPT-1 비교

  |                           |                           GPT-1                           |                  BERT                   |
  | :-----------------------: | :-------------------------------------------------------: | :-------------------------------------: |
  |    Training data size     |                  BookCorpus (800M words)                  | BookCorpus and Wikipedia (2,500M words) |
  |        Batch size         |                       32,000 words                        |              128,000 words              |
  | Task-specific fine-tuning | same learning rate (5e-5) for all fine-tuning experiments | task-specific fine-tuning learning rate |



* BERT로 할 수 있는 대표적인 Task
  * Machine Reading Comprehension (MRC)
    * 기계 독해에 기반한 질의응답(Question Answering)
    * [SQuAD: Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) : 대표적인 데이터
    * SQuAD 1.1 : 답이 있는 곳의 start 부분과 end 부분을 예측하기 위한 FC layer를 각각 둬서 학습한다.
    * SQuAD 2.0 : 질문에 대한 답이 없는 경우까지 포함해서 학습, 질문에 대해서 먼저 답이 있는지 없는지 `[CLS]` 토큰을 이용하여 체크한 후, 답이 있다면 그 위치를 찾는 방식으로 학습
  * On SWAG
    * [SWAG: A Large-scale Adversarial Dataset for Grounded Commonsense Inference](https://leaderboard.allenai.org/swag/submissions/public)
    * 주어진 문장 다음에 나타날법한 적절한 문장을 선택하는 태스크
    * 각 후보에 대해서 주어진 문장과 concat한 후 BERT를 통해 인코딩하고 나온 `[CLS]` 토큰을 이용해서 logit을 구한다.
* BERT: Ablation Study
  * 모델 사이즈를 더 키울수록 여러 downstream task에 대한 성능이 끊임없이 좋아진다.

## [10강] Other Self-supervised Pre-training Models

### [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

* **Language Models are Unsupervised Multitask Learner**s
* Motivation : [decaNLP](https://decanlp.com/)
  * Multitask Learning as Question Answering
  * 다양한 task들을 다 질의응답 형태로 변환할 수 있다는 연구 사례
* Datasets
  * 학습 데이터셋을 좀 더 좋은 퀄리티에다가 양도 늘렸다. (Reddit 커뮤니티 웹사이트에서 조건적으로 크롤링)
  * Preprocess : (Subword 단위) Byte pair encoding (BPE)
* Modification
  * GPT-1에서 레이어를 더 많이 쌓았다. (pretrain task는 똑같다)
  * Layer normalization의 위치가 옮겨짐
  * 레이어를 random initialization할 때, 레이어의 인덱스에 반비례하게 더 작은 값으로 초기화되도록 한다. (위쪽 레이어들의 역할이 줄어들 수 있도록)
* 여러 downstream task가 language 생성 task에서의 (fine-tuning 과정 없이) zero-shot setting으로 다 다뤄질 수 있다.
  * Conversation question answering dataset(CoQA)
  * Summarization : `TL;DR`을 만나면 그 앞쪽 글을 한 줄 요약한다.
  * Translation : 마찬가지로 번역하고 싶은 문장 뒤에 `in French`이런 문구를 넣어주면 번역한다.
* 예시 : [How to Build OpenAI’s GPT-2: “ The AI That Was Too Dangerous to Release”](https://blog.floydhub.com/gpt2/)

### [GPT-3](https://arxiv.org/abs/2005.14165)

* **Language Models are Few-Shot Learners**
  * Zero-shot : 태스크의 자연어 설명만 주어진 채 답을 예측하는 것
  * One-shot : 태스크 설명과 더불어 하나의 태스크 example을 보여주는 것
  * Few-shot : 태스크의 few example을 보여주는 것
    ![image](https://user-images.githubusercontent.com/35680202/133131557-527ebc0b-3cbb-4c01-a056-d6b36a94b9b9.png)
  * 모델 사이즈를 키울수록 성능의 gap이 훨씬 더 크게 올라간다. (큰 모델을 사용할수록 동적인 적응능력이 훨씬 더 뛰어나다.)
* GPT-2 보다 더 많은 Self-Attention block, 더 많은 파라미터 수, 더 많은 데이터, 더 큰 배치 사이즈

### [ALBERT](https://arxiv.org/abs/1909.11942)

* A Lite BERT for Self-supervised Learning of Language Representations (경량화된 형태의 BERT)
* Factorized Embedding Parameterization
  * $V$ = Vocabulary size (ex. 500)
  * $H$ = Hidden-state dimension (ex. 100)
  * $E$ = Word embedding dimension (ex. 15)
  * 파라미터 수 비교
    * BERT : $V \times H = 500 \times 100$
    * ALBERT : $(V \times E) + (E \times H) = 500 \times 15 + 15 \times 100$ (dimension을 늘려주는 레이어 하나를 추가함)

* Cross-layer Parameter Sharing
  ![image](https://user-images.githubusercontent.com/35680202/133187651-6b843036-d0ea-4883-b3d9-ed83f559120b.png)
  * Shared-FFN : 레이어들 간에 feed-forward 네트워크의 파라미터만 공유하기
  * Shared-attention : 레이어들 간에 attention 파라미터만 공유하기
  * All-shared : Both of them, 파라미터 수가 제일 작으면서도 성능의 하락폭이 크지 않다.

* Sentence Order Prediction (For Performance)
  ![image](https://user-images.githubusercontent.com/35680202/133188559-881c4eb0-0952-4bdd-9b0a-bc2e438d8e81.png)
  * 동기 : Next Sentence Prediction pretraining task가 너무 쉽다. (실효성이 없다) Negative sampling을 통해서 별도의 문서에서 뽑은 두 문장 간에 곂치는 단어나 주제가 거의 없을 가능성이 크기 때문이다.
  * 방법 : 이어지는 두 문장을 가져와서 두 문장 순서를 바꾼 후 순서가 올바른지 예측하는 것
  * 결과 : 성능이 더 좋아졌다!

### [ELECTRA](https://arxiv.org/abs/2003.10555)

![image](https://user-images.githubusercontent.com/35680202/133189810-d20573ad-d5da-428e-a1f5-1329b1c349c1.png)

* Pre-training Text Encoders as Discriminators Rather Than Generators
* Efficiently Learning as Encoder that Classifies Token Replacements Accurately
* GAN(Generative adversarial network)에서 사용한 아이디어에서 착안하여 두가지 모델이 적대적 학습(Adversarial learning) 진행
  * Generator : BERT와 비슷한 모델로 학습
  * Discriminator : Generator가 예측한 단어가 원래 있었던 단어인지 판별하는 태스크 수행 (각 단어별로 binary classification) -> 이 모델을 pretrain된 모델로 사용하여 downstream task에서 fine-tuning하여 사용한다.

### Light-weight Models

- 경량화 모델들 : 소형 디바이스에서 로드해서 계산 가능!
- [DistillBERT, a distilled version of BERT: smaller, faster, cheaper and lighter, NeurIPS Workshop'19](https://arxiv.org/abs/1910.01108)
  - huggingface에서 발표
  - student model(경량화된 모델)이 teacher model(큰 사이즈 모델)을 잘 모사할 수 있도록 학습 진행 => **지식 증류(Knowledge Distillation)**
- [TinyBERT: Distilling BERT for Natural Language Understanding, Findings of EMNLP’20](https://arxiv.org/abs/1909.10351)
  - 역시 Knowledge Distillation를 사용하여 경량화
  - teacher model의 target distribution을 모사하는 것 뿐만 아니라, **중간 결과물들까지도 유사해지도록 학습을 진행** : 임베딩 레이어, Self-Attention block이 가지는 $W_Q, W_K, W_V$ (Attention matrix), 결과로 나오는 hidden state vector 등
  - dimension간의 mismatch가 있는 경우 차원 변환을 위한 (학습 가능한) FC layer를 하나 둬서 해결!

### Fusing Knowledge Graph into Language Model

- BERT가 언어적 특성을 잘 이해하고 있는지 분석
  - 주어진 문장, 글이 있을 때 문맥을 잘 파악하고 단어들간의 유사도, 관계를 잘 파악한다.
  - 주어진 문장에 포함되어있지 않은 추가적인 정보가 필요할 때는 효과적으로 활용하지 못한다.
- Knowledge Graph : 이 세상에 존재하는 다양한 개념이나 개체들을 잘 정의하고 그들간의 관계를 잘 정형화해서 만들어둔 것
- => 외부 정보를 Knowledge Graph로 잘 정의하고 Language Model과 결합하는 연구가 진행됨
- [ERNIE: Enhanced Language Representation with Informative Entities, ACL'19](https://arxiv.org/abs/1905.07129)
- [KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning, EMNLP'19](https://arxiv.org/abs/1909.02151)

## 최신 NLP 트렌드

* [Open-Domain Question Answering System](https://lilianweng.github.io/lil-log/2020/10/29/open-domain-question-answering.html)
  * [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
  * Open domain : 다양한 주제로 대화 가능 (ex. [Blender Bot 2.0](https://ai.facebook.com/blog/blender-bot-2-an-open-source-chatbot-that-builds-long-term-memory-and-searches-the-internet/))
  * Closed domain : 특정 주제에 대해서 대화 가능 (ex. 금융 비서)
* [Unsupervised Neural Machine Translation](https://arxiv.org/abs/1710.11041)
* [Text Style Transfer](https://blog.diyaml.com/teampost/Text-Style-Transfer/)
  * [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://arxiv.org/abs/1905.05621)
* Quality Estimation
  * [DeepQuest: a framework for neural-based quality estimation](https://aclanthology.org/C18-1266.pdf)
  * [TransQuest: Translation Quality Estimation with Cross-lingual Transformers](https://arxiv.org/abs/2011.01536)
  * [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
* Large-scale Language Models
  * [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
  * [oLMpics -- On what Language Model Pre-training Captures](https://arxiv.org/abs/1912.13283)
* Transfer Learning
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* In-Context Learning
  * [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* Prompt Tuning
  * [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
  * [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
* Language Models Trained on Code
  * [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
* Multi-Modal Models
  * [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)
  * [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

