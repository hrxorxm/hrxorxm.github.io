---
layout: post
title:  "[통계적 기계학습] 3. Linear Classifiers"
subtitle:   "선형 분류기의 수학적 정의와 종류를 알아보자 (Self-check 有)"
categories: "DeepLearning"
tags: [LinearClassifier]
---

# 선형 분류기

- 목차
    - [1. 용어](#1-용어)
    - [2. Linear Classification](#2-linear-classification)
    - [3. Linear Classifier를 해석하는 세 가지 관점](#3-linear-classifier를-해석하는-세-가지-관점)
      - [(1) Algebraic Viewpoint](#1-algebraic-viewpoint)
      - [(2) Visual Viewpoint](#2-visual-viewpoint)
      - [(3) Geomertic Viewpoint](#3-geomertic-viewpoint)
    - [4. Loss function](#4-loss-function)
      - [용어](#용어)
      - [수학적 정의](#수학적-정의)
      - [Loss function 1 : multiclass SVM loss](#loss-function-1--multiclass-svm-loss)
      - [Loss function 2 : softmax + cross-entropy loss](#loss-function-2--softmax--cross-entropy-loss)
    - [5. Regularization](#5-regularization)
    - [Self-check](#self-check)

## 1. 용어
* Classification : 분류, x,y가 주어지고 y가 순서를 구분할 수 없는 범주형일 때, y를 예측하는 것
* Classifier : 분류기

* `Linear Classifier` : 분류기를 **x의 선형함수**로 만드는 것
  * $Wx + b$ 를 이용해서 <u>범주형 y</u>를 예측하는 것
  * Neural Network로 가는 첫걸음
* `Linear Regression`
  * $wx + b$ 를 이용해서 <u>연속형 y</u>를 예측하는 것
    * $y$ ~ $wx + b$ : y와 x가 일차함수의 관계에 있을 것이라고 추측
  * $\hat{y_i} = wx_i + b$
    * $x_i$ 에 대한 예측값
    * $y_i$ 와 구분하기
  * $\hat{w}, \hat{b} = argmin_{w, b} (\sum_{i=1}^{N}{(y_i - \hat{y_i})^2}) = argmin_{w, b} (\sum_{i=1}^{N}{(y_i - (w x_i + b))^2})$
  * $\hat{w}, \hat{b} = argmin_{w, b} (\sum_{i=1}^{N}{\mid y_i - \hat{y_i} \mid}) = argmin_{w, b} (\sum_{i=1}^{N}{\mid y_i - (w x_i + b)\mid})$
 
## 2. Linear Classification
* Linear Regression과의 차이점
  * $c.g.$ Regression : $x \in R^D$, $wx + b \in R$
  * Classification : $x \in R^D$, $Wx + b \in R^C$ ($C$는 class number)
    * $W = \begin{bmatrix}w1 \\ w2 \end{bmatrix}$, $b = \begin{bmatrix}b1 \\ b2 \end{bmatrix}$
    * $\hat{y_i} = argmax_{k} (w_k x_i + b_k)$

* Parametric Approach
  * $x_i \in R^{32 \times 32 \times 3}$
  * $f(x,W) = Wx + b \in R^{10}$
    * $f(x,W)$ : $(10,)$
    * $W$ : $(10, 3072)$
    * $x$ : $(3072,)$
    * $b$ : $(10,)$
  * $W$ : learnable weights
  * $b$ : learnable bias

## 3. Linear Classifier를 해석하는 세 가지 관점
### (1) Algebraic Viewpoint
* `Bias Trick`
  * 아이디어
    * $f(x,W) = W \cdot x + b \cdot 1$
    * 데이터 벡터 x에 애초에 1이 있었다고 생각하면 weight matrix에 W와 b를 같이 놓을 수 있다.
  * $f(x,W) = W^* x$
  * $W^* = \begin{bmatrix}W & b\end{bmatrix}$
  * $x^* = \begin{bmatrix}x \\ 1\end{bmatrix}$
  * 장점 : 간단하게 표현 가능
    * ex) Linear Classifier의 성질 : $f(x,W) = Wx$ (ignore bias) 일 때, $f(cx,W) = W(cx) = c * f(x,W)$

### (2) Visual Viewpoint
  * 아이디어
    * $W$ 가 필터처럼 들어가고 이미지와 내적 연산
    * 두 벡터의 컴포넌트들이 상수배가 될 수록 내적 연산의 값이 높아진다.
  * 각 클래스에 대응되는 특정 weight 하나를 그림으로 표현
  * **Linear Classifier has One `template per category`**

  * [참고](https://datascienceschool.net/02%20mathematics/02.02%20%EB%B2%A1%ED%84%B0%EC%99%80%20%ED%96%89%EB%A0%AC%EC%9D%98%20%EC%97%B0%EC%82%B0.html)
    * 벡터-벡터 곱셈 (내적(inner product))
      * $x \cdot y = < x, y > = x^T y$
      * 두 벡터의 차원(길이)이 같아야 한다.
      * 앞의 벡터가 행 벡터이고 뒤의 벡터가 열 벡터여야 한다.
    * 행렬-행렬 곱셈
      * $A \in R^{N \times L}$, $B \in R^{L \times M}$ $\rightarrow$ $AB \in R^{N \times M}$
      * $C = AB$ $\rightarrow$ $c_{ij} = a_{i}^{T} b_{j}$
      * 원소 $c_{ij}$ 의 값은 $A$ 행렬의 $i$ 번째 행 벡터와 $B$ 행렬의 $j$ 번째 열 벡터의 곱이다.

### (3) Geomertic Viewpoint
  * $f(x,W) = Wx + b$
    * $W$ : (3, 3072)
    * $x$ : (3072, )
    * $b$ : (3, 1)
  * $W = \begin{bmatrix}w_1 \\ w_2 \\ w_3\end{bmatrix}$, $b = \begin{bmatrix}b_1 \\ b_2 \\ b_3\end{bmatrix}$
    * $w_i$ : (1, 3072)
    * $b_i$ : (1, 1)
  * $w_i x + b_i = 0$ : `초평면` (3072-dimensional Euclidean space)
  * 일반화
    * $f_k(x) = w_k x + b_k$
  * `Hyperplanes cutting up space`
  * decision boundary : Linear classifier는 결정 경계가 언제나 선형이다.


## 4. Loss function
### 용어
* 주어진 $W, b$ 가 얼마나 좋은지 평가
  * loss가 낮을 수록 좋은 classifier
  * loss가 높을 수록 나쁜 classifier
* 비슷한 용어
  * Loss function : 손실함수 (보통 하나의 input data 에 대해서 오차를 계산한 것)
  * Cost function : 비용함수 (보통 모든 input data 에 대해서 오차를 계산한 것)
  * Objective function : 목적함수 (여기서는 최소화하고 싶은 함수)
    * loss (data loss) + regularizer (regularizer loss) : loss의 최소화를 regularizer가 규제함
    * loss 와 regularizer 는 Objective function 의 Component
* 반대 용어 (Negative loss function)  
  * reward function
  * profit function
  * utility function : 효용함수
  * fitness function

### 수학적 정의
* dataset
  * $\{(x_i,y_i)\}^{N}_{i=1}$
  * $x_i$ : image
  * $y_i$ : (integer) label $\in (1, \dotsc, C)$
* Loss for a single example
  * $L_i(f(x_i, W), y_i)$
* Loss for the dataset
  * $\phi(W) = \frac{1}{N} \sum_{i=1}^{N} L(f(x_i, W), y_i)$
  * 각 example의 loss에 대한 평균

#### Regression 에서의 정의
* 기본 정의
  * $f(x, w) = wx + b$ ($x \in R^D$, $f(x, w) \in R \mid = \hat{y}$)
  * $L(\hat{y}, y) =$
    * $(\hat{y} - y)^2$ (Squared error loss, 오차제곱 손실함수)
    * $\mid \hat{y} - y \mid$ (Absolute error loss, 절댓값 오차 손실함수)
* Loss for single example (Squared error loss 기준)
  * $L_i(f(x_i, W), y_i) = (y_i - (wx_i + b))^2$
* Objective function
  * $\frac{1}{N} \sum_{i=1}^{N} (y_i - (wx_i + b))^2$
* Let minimizer of $\phi(w, b)$ be $\hat{w}, \hat{b}$. 
  * $(\hat{w}, \hat{b}) = argmin_{w, b} \phi(w,b)$
* when $x_{new}$ comes, we predict $\hat{y}_{new}$ as 
  * ${\hat{y}}_{new} = f(x_{new};\hat{w}, \hat{b}) = \hat{w} x_{new} + \hat{b}$

#### Classification 에서의 정의
* 기본 정의
  * $f(x, W) = Wx + b$ ($x \in R^D$, $f(x, W) \in R^{C_{(클래스 개수)}} \mid = s_{(스코어)}$)
    * $s \in R^C$, $s = \{ s_1, \dotsc, s_C \}^T$
    * $y \in \{1 \dotsc C\}$
  * Loss function (multiclass svm loss)
    * $L(s, y) = \sum_{j \neq y} \max (0, s_j - s_y + 1)$
  * Loss function (softmax + cross-entropy loss : logistic regression 의 loss function)
    * $L(s, y) = - log (\frac{e^{s_y}}{\sum_{j=1}^{C} e^{s_j}})$
* Loss for single example
  * $s_i (\in R^C) = (s_{i1}, \dotsc, s_{iC}) = f(x_i ; w) = wx_i + b$
  * $L_i(f(x_i;W), y_i)$
* Objective function
  * 편의상, $W \in (W, b)$ 라고 하자
  * $\phi(W) = \frac{1}{N} \sum_{i=1}^{N} L_i(f(x_i;W), y_i) = \frac{1}{N} \sum_{i=1}^{N} L_i(W x_i + b, y_i)$
  * $(\hat{w}, \hat{b}) = argmin_{w, b} \phi(w,b)$
* when $x_{new}$ comes, we predict $\hat{y}_{new}$, which is defined by
  * $s_{new} = \hat{W}x_{new} + \hat{b} = (s_{new1}, s_{new2}, \dotsc, s_{newC})$
    * $\hat{W}$ : $C \times D$
    * $x_{new} \in R^D$
    * $\hat{b}$ : $C \times 1$
    * $s_{new} \in R^{C \times 1}$
  * ${\hat{y}}_{new} = argmax_{j \in \{ 1, \dotsc, C \}} {(s_{new})}_j$

### Loss function 1 : multiclass SVM loss
* 용어
  * `Hinge Loss` : 경첩 손실 함수
  * `Margin` : "loss가 0이 되는 지점의 score" 와 "나머지 class에서 가장 높은 score" 의 차이

* 수학적 정의
  * 되도록 $s_y > s_j (j \neq y)$ 이 되기를 바란다.
  * single example $(x_i, y_i)$ 일 때, $s = f(x_i, W) = (s_1, s_2, \dotsc, s_C)$ 라고 하자.
  * $L_i = \sum_{j \neq y_i} \max (0, s_j - s_{y_i} + 1)$

* 대수적인 성질 (6 properties)
  1. 다른 class에 비해 정답 score 가 높아서 loss가 0 이었던 (margin 이 큰 상황인) 이미지 픽셀이 조금 변한다면?
     * 그래도 웬만하면 loss가 그대로 0 이 나올 것이다.
  2. loss 값으로 가능한 최솟값과 최댓값은?
     * min loss : 0
     * max loss : $+ \infty$
  3. score 값이 랜덤 (standard normal로 뽑는다면) 이라면 loss 값이 어떻게 나올까?
     * 모든 N에 대해서 평균적으로 $s_j - s_{y_i} \approx 0$ 라고 생각할 수 있고,
     * $\frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max (0, s_j - s_{y_i} + 1) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max (0, 1) = C - 1$
  4. 만약 $\sum_{j \neq y_i}$ 가 아니라 $\sum_{j=1}^{C}$ 로 계산한다면 값이 어떻게 될까?
     * $\max (0, s_{y_i} - s_{y_i} + 1) = \max(0, 1) = 1$
     * 기존 loss 에 상수 1 만 더한 값
  5. 만약 Sum ($\sum_{j \neq y_i}$) 이 아니라 Mean ($\frac{1}{C-1} \sum_{j \neq y_i}$) 으로 계산한다면 값이 어떻게 될까?
     * 상수배한다고 해도, 대소관계가 바뀌는 것은 아니고, 어차피 loss를 최소화 시키는 것이 목적이기 때문에 최소화 문제라는 것이 바뀌지는 않을 것이다.
  6. 만약 $\sum_{j \neq y_i} \max (0, s_j - s_{y_i} + 1)$ 이 아니라 $\sum_{j \neq y_i} \max (0, s_j - s_{y_i} + 1)^2$ 으로 계산한다면 값이 어떻게 될까?
     * 아예 새로운 loss를 정의한 것
     * 더 이상 Multiclass SVM Loss가 아니다.

* 추가적인 고민
  * $f(x, W) = Wx (\in R^{C \times 1})$ : Bias Trick
  * $\phi (W) = L = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max (0, f{(x_i;W)}_{j} - f{(x_i;W)}_{y_i} + 1)$
    * $W$ 에 대한 함수로 볼 수 있다.
  * $L = 0$ 로 만드는 $W$ 를 찾았을 때, $W$ 가 unique 할까?
    * No!
    * ex) $cW (c > 0)$, 즉, $W$에 상수배를 한다고 해서 score의 대소관계가 바뀌지 않기 때문에 똑같은 $L$ 값을 갖게 된다.
    * 유일한 최소화원이 있는 문제는 아니다.
  * 따라서, data loss 의 최소화 뿐만 아니라 다른 기준을 추가하여 $W$ 를 선택하게 된다.

### Loss function 2 : softmax + cross-entropy loss
* [용어 및 수학적 정의](https://wikidocs.net/35476)
  * $s = f(x_i;W)$
  * `Softmax function` : ${softmax(s)}_{k} = P(Y = k \mid X = x_i) = \frac{e^{s_k}}{\sum_{j} e^{s_j}}$
    * 기존 : $L(s, y)$, $s \in R^C$
    * 변경 : $L(t, y)$, $t = \frac{1}{\sum_{j} e^{s_j}} e^{s} \in R^C$
      * $0 < t_j < 1$
      * $\sum_{j=1}^{C} t_j = 1$
  * `Cross-Entropy Loss (Multinomial Logistic Regression)`
    * $L(t, y) = - log (t_y)$
    * correct class 의 확률의 마이너스 로그값
    * $t_y$ 값이 $1^-$에 가까울 수록, $L(t, y)$ 값은 $0^+$ 에 가까워진다.
    * $t_y$ 값이 $0^+$에 가까울 수록, $L(t, y)$ 값은 $+\infty$ 에 가까워진다.
  * $L(s, y) = - log (\frac{e^{s_y}}{\sum_{j=1}^{C} e^{s_j}})$
  * $L_i = - log P(Y = y_i \mid X = x_i) = - log (\frac{e^{s_{y_i}}}{\sum_{j} e^{s_j}})$
* (참고)
  * `Maximum Likelihood Estimation` ?
  * `Kullback-Leibler divergence` ?

* 대수적인 성질
  1. loss 값으로 가능한 최솟값과 최댓값은?
     * min loss : $0^+$
     * max loss : $+ \infty$
  2. score 값이 (small) 랜덤값 (standard normal로 뽑는다면) 이라면 loss 값이 어떻게 나올까?
     * 모든 $i$ 에 대해서 평균을 내면 : $- log (\frac{1}{C})$
     * ex) $C = 10$ 일 때, $- log(\frac{1}{10}) = log(10) \approx 2.3$

### Comparison of SVM and Cross-entropy losses
* 이론적인 성질이 아니라 경험에서 온 직관으로서, **Cross-entropy loss 를 더 선호**하는 이유
  1. SVM loss 에는 확률적 해석이 없지만, Cross-entropy loss 에는 **확률적 해석이 있다.** 가끔 확률적 해석이 많이 쓰일 때가 있다.
  2. 둘이 경험상 큰 차이가 없다.

* 대수적인 성질 비교
  * 상황 : 모든 레코드에서 correct class 의 score 가 가장 높다. (**perfect separation**)
    * 그저 과학적인 비교일 뿐, 현실에서 잘 일어나지 않는 상황임
    * 성능을 개선시킬 포인트가 loss 외에 더 많기 때문에 어떤 loss function 을 쓸지 고민하는 것보다, 데이터를 더 깨끗하게 정제하거나 네트워크를 어떻게 쌓을지 등을 더 고민하는 것이 좋을 것이다.

  > Q. 이 때, Cross-entropy loss 와 SVM loss 의 차이는?
  * Cross-entropy loss 는 0 에 가까워지지만, SVM loss 는 정확히 0 (exact zero) 이 된다는 장점이 있다.

  > Q. correct class 의 score 를 더 높인다면?
  * Cross-entropy loss 는 조금 더 0 에 가까워 질 것이고, SVM loss 는 여전히 0 일 것이다.

  > Q. correct class 가 아닌 class 의 score 를 살짝 바꾼다면?
  * Cross-entropy loss 는 조금 변하지만, SVM loss 는 변하지 않을 것이다.
  * SVM loss 는 이미 잘 하고 있는 문제에 대해서는 더 이상 케어하지 않는다.
  * Cross-entropy loss 는 이미 잘 하고 있는 상황에서도 더 잘하기 위해 minimize 를 계속 하려고 할 것이다.

## 5. Regularization
* 용어
  * Regularization : 정규화, **규제** (training 과정을 방해하고 규제하는 역할을 한다.)

* Regularization : Beyond Training loss
  * `Full loss` : $L(W) = \frac{1}{N} \sum_{i=1}^{N} L_i (f(x_i, W), y_i) + \lambda R(W)$
    * `Data loss` : $\frac{1}{N} \sum_{i=1}^{N} L_i (f(x_i, W), y_i)$
      * $\hat{y}_i \approx y_i$ on training data 를 원한다.
      * $\hat{y}_i = y_i ( \forall i )$ (모든 $i$ 에 대해서 예측값이 실제값과 같다면, Data loss 가 항상 최소가 될 수 있다.
      * 현실에서 $y_i$ 에 노이즈가 많이 껴있기 때문에 항상 예측값과 같기는 어렵다. (관찰한 것만으로는 설명할 수 없는 많은 잡음들이 껴있기 마련, 또는 아웃라이어)
    * `Regularization` : $\lambda R(W)$
      * 모델이 training data에 너무 정확하게 맞는 것을 방해한다.
      * 오직 $W$ 만 가지고 계산한다.
      * `라그랑즈 승수법` ?
        * $L(W)$ 를 최소화 시키는 것은, $\iff_{(수학적 동치)}$ $R(W) \leq c_{(상수)}$ 로 **제한되어 있는 상황에서** $\min \frac{1}{N} \sum_{i=1}^{N} L_i$ 을 구하는 문제와 같다.
        * Simple examples
          * L2 regularization : $R(W) = \sum_{k} \sum_{l} W_{k,l}^{2}$
          * L1 regularization : $R(W) = \sum_{k} \sum_{l} \mid W_{k,l} \mid$
          * Elastic net (L1 + L2) : $R(W) = \sum_{k} \sum_{l} (\beta W_{k,l}^{2} + \mid W_{k,l} \mid)$
  * 다른 방법으로 규제하는 기법
    * `Dropout` : 중간에 weight 를 0 으로 날려버리는 것
    * `Batch normalization` : 중간 중간 normalization 을 다시 하는 것
    * `Cutout, Mixup, Stochastic depth, etc...`
    * 일반적으로 Batch normalization 과 L2 regularization 을 같이 자주 쓰는 것 같다.
  * 목적
    * Expressing Preferences : Data loss 만 최소화 시키는 것이 아니라, 모델의 다른 선험적인 제약조건을 주입하는 것, 어떤 모델을 더 선호하겠다는 것을 주입하는 것
      * ex) L2 regularization likes to "spread out" the weights
      * weight 가 모든 class 에 고르게 퍼져있는 것이 안정적인 classifier에 도움이 된다고 한다.
    * Prefer Simpler Models : $y_i$ 와 $\hat{y}_i$ 이 너무 완벽하게 같아지는 것을 피하고 싶다. 학습 데이터를 모델이 너무 다 설명하는 것을 **Overfitting (과대적합)** 이라고 부른다.
    * curvature (곡률) 을 추가함으로써 최적화 자체를 도울 수 있다.
      * ex) L2 규제 : 2차항을 더한다는 뜻 -> 목적함수의 2nd derivative 가 최대한 positive definite (양의 값) 에 가까울 수록, 목적함수가 $W$ 에 볼록한 함수처럼 생겨서 최솟값을 찾기가 더 쉬워진다.


# Self-check
* 데이터와 지도학습에 기반한 다범주(multiclass) 분류법을 소개하고자 합니다.

## 문제1
* 선형변환된 score, cross-entropy loss 최소화, no regularization 기반한 분류기 구축법을 소개하여 주세요 ("training dataset을 (x_i, y_i), i=1,…,N 으로 나타내자."부터 시작하세요.) - 1 page
* 답
  * training set : $\{(x_i,y_i)\}^{N}_{i=1}$
    * $X : (N, D)$
    * $y : (N, )$
  * 선형변환된 score
    * $W : (D, C)$, $b : (C, )$
    * $S = XW + b : (N, C)$
  * loss function : cross-entropy loss 최소화 (no regularization)
    * correct class 의 확률의 마이너스 로그값
    * $L_i = - log(P(Y = y_i \mid X = x_i)) = - log(\frac{exp(S_{i y_i})}{\sum_{i=1}^{N} exp(S_{ik})})$
      * $P(Y = y_i \mid X = x_i)$ 값이 $1^-$에 가까울 수록, $L_i$ 값은 $0^+$ 에 가까워진다.
      * $P(Y = y_i \mid X = x_i)$ 값이 $0^+$에 가까울 수록, $L_i$ 값은 $+\infty$ 에 가까워진다.
    * $L = \frac{1}{N} \sum_{i=1}^{N} (-log(\frac{exp(S_{i y_i})}{\sum_{i=1}^{N} exp(S_{ik})}))$

## 문제2
* 1번에서 cross-entropy loss를 multiclass SVM loss로 변경하고자 합니다. 1번 답안에서 변경되는 부분은 무엇입니까? - 0.25 pages 
* 답
  * loss function : multiclass SVM loss 로 변경
    * $L_i = \sum_{j \neq y_i} \max (0, s_j - s_{y_i} + 1)$
    * $L = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max (0, s_j - s_{y_i} + 1)$
  * 차이점
    * Cross-entropy loss 는 0 에 가까워지지만, SVM loss 는 정확히 0 (exact zero) 이 된다는 장점이 있다.
    * correct class 의 score 를 더 높인다면, cross-entropy loss 는 조금 더 0 에 가까워 질 것이고, SVM loss 는 여전히 0 일 것이다.
    * correct class 가 아닌 class 의 score 를 살짝 바꾼다면, cross-entropy loss 는 조금 변하지만, SVM loss 는 변하지 않을 것이다. SVM loss 는 이미 잘 하고 있는 문제에 대해서는 더 이상 케어하지 않는다. cross-entropy loss 는 이미 잘 하고 있는 상황에서도 더 잘하기 위해 minimize 를 계속 하려고 할 것이다.

## 문제3
* 1번에서 regularization을 첨가하고자 합니다.  1번 답안에서 변경되는 부분은 무엇입니까? regularization의 예시는 무엇이 있습니까? regularization은 왜 하나요? - 0.5 pages
* 답
  * loss function : cross-entropy loss 최소화 + regularization 추가
    * $L = \frac{1}{N} \sum_{i=1}^{N} (-log(\frac{exp(S_{i y_i})}{\sum_{i=1}^{N} exp(S_{ik})})) + \lambda R(W)$
    * regularization의 예시
      * L2 regularization : $R(W) = \sum_{k} \sum_{l} W_{k,l}^{2}$
      * L1 regularization : $R(W) = \sum_{k} \sum_{l} \mid W_{k,l} \mid$
      * Elastic net (L1 + L2) : $R(W) = \sum_{k} \sum_{l} (\beta W_{k,l}^{2} + \mid W_{k,l} \mid)$
    * 목적
      * Expressing Preferences : Data loss 만 최소화 시키는 것이 아니라, 모델의 다른 선험적인 제약조건을 주입하는 것, 어떤 모델을 더 선호하겠다는 것을 주입하는 것
        * ex) L2 regularization likes to "spread out" the weights
        * weight 가 모든 class 에 고르게 퍼져있는 것이 안정적인 classifier에 도움이 된다고 한다.
      * Prefer Simpler Models : $y_i$ 와 $\hat{y}_i$ 이 너무 완벽하게 같아지는 것을 피하고 싶다. 학습 데이터를 모델이 너무 다 설명하는 것을 **Overfitting (과대적합)** 이라고 부른다.
      * curvature (곡률) 을 추가함으로써 최적화 자체를 도울 수 있다.
        * ex) L2 규제 : 2차항을 더한다는 뜻 -> 목적함수의 2nd derivative 가 최대한 positive definite (양의 값) 에 가까울 수록, 목적함수가 $W$ 에 볼록한 함수처럼 생겨서 최솟값을 찾기가 더 쉬워진다.