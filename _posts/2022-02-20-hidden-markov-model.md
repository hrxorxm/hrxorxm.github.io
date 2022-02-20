---
layout: post
title:  "[핵심 머신러닝] Hidden Markov Models"
subtitle:   "Sampling, STFT, Spectrogram, Mel-Spectrogram, MFCC 등에 대한 강의 내용 정리"
categories: "Speech-Recognition"
tags: []
use_math: true
---

# Hidden Markov Model

> 강의 출처 : [[핵심 머신러닝] Hidden Markov Models - Part 1 (개념, Evaluation)](https://youtu.be/HB9Nb0odPRs), [[핵심 머신러닝] Hidden Markov Models - Part 2 (Decoding, Learning)](https://youtu.be/P02Lws57gqM)

## 개요

* 순차 데이터 (Sequential data) : 시간적 특성이 있음
* Hidden Markov Model (HMM)
  * 순차 데이터를 확률적(Stochastic)으로 모델링하는 생성 모델(Generative Model)
  * Hidden : 숨겨져 있는 (관측이 되지 않는)
  * Markov Model : 마코프 모델

## Markov Model

### 개념

* State로 이루어진 Sequence를 상태 전이 확률 행렬로 표현하는 것
  * 상태 전이 확률 행렬 (State transition probability matrix)
  * 예시)
    * State : {비, 해}
    * Sequence : 비 -> 해 -> 해 -> 해 -> 비 -> 비 -> 해
    * State transition probability matrix : From {비, 해} -> To {비, 해} (2by2 행렬)
      <img src="https://user-images.githubusercontent.com/35680202/154851222-e6ade05a-ec36-4722-a2ae-b130f10970f9.png" width="200" height="100" />
* Markov 가정 : 시간 t에서 관측은 가장 최근 r개의 관측에만 의존한다는 가정
  * 한 상태에서  다른 상태로의 전이는 이전 상태의 긴 이력을 필요치 않는다는 다정
  * if r = 1 : First order Markov model
    * $p(s_t \| s_{t-1}, s_{t-2}, ... ,s_1) = p(s_t \| s_{t-1})$
  * if r = 2 : Second order Markov model
    * $p(s_t \| s_{t-1}, s_{t-2}, ... ,s_1) = p(s_t \| s_{t-1}, s_{t-2})$

### 파라미터

* 상태 전이 확률 행렬 $A(a_{ij})$
  <img src="https://user-images.githubusercontent.com/35680202/154851335-7a72a4c8-2989-43ce-babd-a445df938925.png" width="400" height="150" />

## Hidden Markov Model

### 개념

* 같은 시간에 발생한 두 종류의 state sequence 각각의 특성과 그들의 관계를 이용하여 모델링
  <img src="https://user-images.githubusercontent.com/35680202/154851369-076cde1c-6049-46f1-9ee1-c05196bbc7ef.png" width="600" height="150">
  * Type1 sequence는 숨겨져 있고(Hidden), Type2 sequence는 관측이 가능(Observable)
  * Hidden sequence 가 Markov assumption을 따름 -> 순차적 특성을 반영
  * Observable sequence 는 순차적 특성을 반영하는 Hidden sequence 에 종속
* Hidden Markov Model
  <img src="https://user-images.githubusercontent.com/35680202/154851286-c814bb5f-e833-4f18-8351-7dcad39cde07.png" width="600" height="300">

### 파라미터

* $A(a_{ij})$ : **State transition probability matrix** (상태 전이 확률 행렬)
  * HMM 이 작동하는 도중 다음 상태를 결정
  * $a_{ij} = P(q_{t+1}=s_j \| q_t = s_i), 1 \leq i,j \leq n$
  * $\sum_{j=1}^{n} a_{ij} = 1$
* $B(b_{jk})$ : **Emission probability matrix** (방출 확률 행렬)
  * HMM 이 어느 상태에 도달하였을 때, 그 상태에서 관측될 확률 결정
  * $b_j(v_k)$ : 은닉상태 $b_j$에서 관측치 $v_k$가 도출될 확률
  * $b_j(v_k) = P(o_t=v_k \| q_t = s_j), 1 \leq j \leq n, 1 \leq k \leq m$
  * $\sum_{j=1}^{n} b_j(v_k) = 1$
  <img src="https://user-images.githubusercontent.com/35680202/154851417-fe5fa16b-15c8-4492-a1f5-70acd736683a.png" width="200" height="200">
* $\pi(\pi_{i})$ : **Initial state probability matrix** (초기 확률 행렬)
  * HMM 을 가동시킬 때 어느 상태에서 시작할지 결정
  * $\pi_i$ : $s_i$에서 시작할 확률
  * $\sum_{i=1}^{n} \pi_i = 1$
  <img src="https://user-images.githubusercontent.com/35680202/154851462-50e11a67-11f3-4ab0-b034-ab685ffd187a.png" width="400" height="200">

### HMM의 주요 문제

* **Evaluation problem** : 파라미터 값들을 추청한 HMM model, 즉, HMM($\lambda *$) 와 Observable state sequence $O$가 주어졌을 때, $O$의 확률 찾기
  * 문제 : hidden state의 총 상태 개수가 N, sequence 길이가 T일 때, 총 경우의 수 = $N^T$
    * 예시) N=3, T=20 일 때, 35억 가지의 경우의 수가 생김
  * 해결 : **Forward algorithm**, **Backward algorithm**
    * Probability_Forward(O) = Probability_Backward(O)
    * Forward probability
      * $\alpha_t(i) = [\sum_{j=1}^{n} \alpha_{t-1}(j) \cdot a_{ji}] \cdot b_i(o_t), 2 \leq t \leq T, 1 \leq i \leq n$
      * $\alpha_{t=1}(i) = \pi_i b_i(o_{t=1})$ : 초기 시점
    * Backward probability
      * $\beta_t(i) = [\sum_{j=1}^{n} \beta_{t+1}(j) \cdot a_{ij}] \cdot b_j(o_{t+1}), 1 \leq t \leq T-1, 1 \leq i \leq n$
      * $\beta_{t=T}(i) = 1$ : 마지막 시점
* **Decoding problem** : HMM($\lambda *$) 와 $O$가 주어졌을 때, 거기에 해당하는 optimal hidden state sequence $S$ 찾기 (HMM의 핵심!)
  * 문제 : 관측 상태 시퀀스가 주어졌을 때 가장 그럴싸한 은닉 상태 시퀀스 결정하기
  * 해결 : **Viterbi algorithm**
    * $v_t(i)$ : Viterbi확률, $t$ 번째 시점의 $i$ 은닉상태의 확률
      * $v_t(i) = \underset{q_1, q_2, ..., q_{t-1}}{\max} p(o_1, o_2, ..., o_t, q_1, q_2, ..., q_{t-1}, q_t = s_i \| \lambda) = [\underset{1 \leq j \leq n}{\max} v_{t-1}(j) a_{ji}] \cdot b_i(o_t), 2 \leq t \leq T, 1 \leq i \leq n$
      * $v_{t=1}(i) = \pi_i b_i(o_{t=1})$
    * $\hat{Q}_t = (\hat{q}_1, \hat{q}_2, ..., \hat{q}_t)$ : 찾고자하는 hidden state sequence
      * $\hat{q}_T = \underset{1 \leq j \leq n}{argmax}$ $v_T(j)$
      * $\hat{q}_t = \tau_{t+1}(\hat{q}_{t+1}), t=T-1,T-2,...,1$
        * $\tau_t(i) = \underset{1 \leq j \leq n}{argmax} [v_{t-1}(j) a_{ji}], 2 \leq t \leq T, 1 \leq i \leq n$ : t-1 시점의 어떤 은닉 상태(j)로부터 t 시점의 은닉상태(i)가 결정되었는지
* **Learning problem** : Observable state sequence $X = \{ O_1, ..., O_N \}$ 이 주어졌을 때, HMM($\lambda *$) 찾기(parameter estimation)
  * 문제 : 관측 상태 시퀀스 O의 확률을 최대로하는 HMM parameter 를 찾자 (여러 개의 관측 시퀀스를 줄테니 최적의 HMM parameter 를 찾자)
    * HMM($\lambda *$) = $\underset{\lambda}{argmax} P(O \| \lambda)$
  * 해결 : **Baum-Welch algorithm** (or **forward-backward algorithm**)
    * **EM(Expectation-maximization) algorithm** 에 속함
      * E-step : hidden state estimation -> $\alpha, \beta$ 를 계산하여 $\gamma_t(i), \xi_t(i,j)$ 를 구하는 것
      * M-step : HMM($\lambda$) update -> $\gamma_t(i), \xi_t(i,j)$ 를 이용하여 HMM($\lambda$) 개선 -> HMM($\lambda^{new}$)
    * $\gamma_t(i)$ : HMM($\lambda$), O 가 주어졌을 때, t 시점 상태가 $s_i$ 일 확률
      * $\alpha_t(i) = p(o_1, o_2, ..., o_t, q_t = s_t \| \lambda) = [\sum_{j=1}^{n} \alpha_{t-1}(j) a_{ji}] \cdot b_i(o_t)$ : Forward probability
      * $\beta_t(i) = p(o_{t+1}, o_{t+2}, ..., o_T \| q_t = s_i, \lambda) = [\sum_{j=1}^{n}a_{ij}b_i(o_t)\beta_{t}(j)]$ : Backward probability
        * $\alpha_t(i) \beta_t(i)$ : t 번째 시점에 상태 i 를 지나는 모든 경로에 해당하는 확률의 합
      * $\gamma_t(i) = p(q_t=s_i \| O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^{n} \alpha_t(j) \beta_t(j)}, 1 \leq t \leq T, 1 \leq i \leq n$
        * ($s_i$ 일 확률) / ($s_{1 \sim n}$ 일 확률 모두 더한 값
    * $\xi_t(i, j)$ : HMM($\lambda$), O 가 주어졌을 때, t 시점 상태가 $s_i$, t+1 시점 상태가 $s_j$ 일 확률
      * $\xi_t(i,j) = \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{\sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_t(i) a_{ij} b_j(o_t{t+1}) \beta_{t+1}(j)}, 1 \leq t \leq T-1, 1 \leq i,j \leq n$
        * $a_{ij} b_j(o_{t+1})$ : i 상태와 j 상태 연결
    * 파라미터 업데이트 방식
      * $\pi_i^{new} = \gamma_{t=1}(i)$ : t 가 1 일 때 $s_i$ 에 있을 확률 ($1 \leq i \leq n$)
      * $a_{ij}^{new} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$ : ($s_i$에서 $s_j$로 전이할 기대값) / ($s_i$에서 전이할 기대값)  ($1 \leq i,j \leq n$)
        * $\sum_{t=1}^{T-1} \gamma_t(i) = \sum_{t=1}^{T-1} \sum_{k=1}^{N} \xi_t(i,k)$
      * $b_i(v_k)^{new} = \frac{\sum_{t=1, st. o_t=v_t}^{T} \gamma_t(i)}{\sum_{t=1}^{T} \gamma_t(i)}$ : ($s_i$에서 $v_k$를 관측할 확률) / ($s_i$에 있을 확률) ($1 \leq i \leq n, 1 \leq k \leq m$)
        * st : such that (뒤는 조건을 의미)

### HMM의 응용

* Evaluation 예시 : 제조 공정 저수율 설비 경로 탐지
  * 관측 시퀀스 : 설비 ID 데이터
  * 은닉 시퀀스 : 공정 ID 데이터
  * 클래스 : 정상 or 불량
  * 예측 : 정상 데이터로 학습한 HMM(정상) 모델과 불량 데이터로 학습한 HMM(불량) 모델에 새로운 관측 시퀀스를 넣은 후 더 큰 확률값을 가지는 클래스로 예측
    * $\underset{class}{argmax} P(O_{3001} \| HMM(class) ... class = \{정상 or 불량\})$
* Decoding 예시 : Sleep Quality Evaluation
  * 관측 시퀀스 : 수면 EEG 데이터
  * 은닉 시퀀스 : REM-수면, NREM-수면, Wake
