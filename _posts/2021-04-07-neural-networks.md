---
layout: post
title:  "[통계적 기계학습] 5. Neural Networks"
subtitle:   "신경망과 비선형 변환의 수학적 정의와 성질에 대해 알아보자 (Self-check 有)"
categories: "Statistical-Machine-Learning"
tags: []
use_math: true
---

# Neural Networks

- 목차
    - [목표](#목표)
    - [Traditional approach : Feature transform + linear classifier](#traditional-approach--feature-transform--linear-classifier)
    - [Fully-connected neural network](#fully-connected-neural-network)
    - [Why Neural Network?](#why-neural-network)
    - [A theoretical guarantee from neural networks](#a-theoretical-guarantee-from-neural-networks)
    - [Neural networks are seldomly convex](#neural-networks-are-seldomly-convex)
    - [Self-check](#self-check)

## 목표
* Problem : Linear Classifiers aren't that powerful.
  * Geometric Viewpoint 에서, 데이터가 선으로 가를 수 없게 분포되어 있을 수도 있다.
  * Visual Viewpoint 에서, One template per class 는 한 class 의 다른 모드를 인식할 수 없다.
* 해결 : non-linear 한 $f(x;W)$ 를 구한다.
  * Fully-connected neural network
  * = Multi-layer perceptron (MLP)
  * = Feed-forward neural network (FF)

## Traditional approach : Feature transform + linear classifier
* Feature Transforms
  * ex) $X \in R^2$, $X = (x, y)$ 에서 극좌표계 변환 : $r = (x^2 + y^2)^{1/2}$, $\theta = tan^{-1}(y / x)$
  * Nonlinear classifier in original space -> **Linear classifier in Feature space**
  * Image Features
    * Color Histogram
      * Ignores texture, spatial positions
      * 객체가 어디 있던지 상관없이 객체를 안정적으로 뽑아줄 수 있을 것이다.
    * Histogram of Oriented Gradients (HoG)
      * 방향 미분의 히스토그램
      * 질감이나 위치 등이 작게 변하는 것에 둔감하게 피쳐를 뽑아줄 수 있다.
    * Bag of Words (Data-Driven 방식)
      * 그림을 patch 로 자르고, 미리 만들어놓은 codebook 의 각 patch와 유사한 요소의 빈도를 히스토그램으로 나타낸다.
    * 예시
      * $X \in R^D$ 에서
      * 각각 $\Phi_1(X) \in R^{P_1}$, $\Phi_2(X) \in R^{P_2}$, $\Phi_3(X) \in R^{P_3}$ 를 구한 후
      * $\Phi(X) = (\Phi_1(X), \Phi_2(X), \Phi_3(X)) \in R^{P_1 + P_2 + P_3}$ 를 만들어서 선형 분류를 한다.
      * (2011년까지 잘 쓰임)
  * 차이점 : `Neural Networks 는 Feature Extraction 을 하는 방법까지 데이터로부터 배운다.`

## Fully-connected neural network
* $f(x;W)$ 를 만드는 방식
  * (Before) Linear score function
    * $f = Wx$
    * $x \in R^D, W \in R^{C \times D}$
  * (Now) 2-layer Neural Network
    * $f = W_2 max(0, W_1 x)$ 
      * $h = W_1 x$ 일 때, $h_i = {(W_1)}_{11} x_1 + \dotsb + {(W_1)}_{ij} x_j + \dotsb + {(W_1)}_{iD} x_D$
      * $W_{ij}$ 는 $x_j$ 에서 $h_i$ 로의 effect 의 크기를 나타낸다.
      * $W$의 모든 성분이 0이 아니라면, x 의 모든 성분이 h 성분을 만드는데 기여한다.
    * ($f = W_2 \sigma (W_1 x)$ 이고, $\sigma (t) = max (0, t)$)
      * $\sigma$는 non-linear 함수, 인풋을 비선형으로 바꿔주는 역할
      * $\sigma (t_1, \dotsc, t_H) = max(0,t_1), \dotsc$ 라고 생각
    * $W_2 \in R^{C \times H}, W_1 \in R^{H \times D}, x \in R^D$ 
  * or 3-layer Neural Network
    * $f = W_3 max(0, W_2 max(0, W_1 x))$
    * $W_3 \in R^{C \times H_2}, W_2 \in R^{H_2 \times H_1}, W_1 \in R^{H_1 \times D}, x \in R^D$
    * 선형변환 -> non-linear -> 선형변환 -> non-linear -> 선형변환
    * 이를 반복하면 됨
  * Deep Neural Networks
    * Depth = number of layers
    * Width = size of each layer
    * ex) $s = W_6 max(0, W_5 max(0, W_4 max(0, W_3 max(0, W_2 max(0, W_1 x)))))$
    * 합성을 여러 번 할 수 있다. 얼마든지 다양한 구조로 생각할 수 있다.

* Activation Functions (활성함수)
  * Q. 만약 activation function 이 없다면?
    * $s = W_2 W_1 x$ 일 때, $W_3 = W_2 W_1 \in R^{C \times H}$ 라고 정의하면, $s = W_3 x$ 가 된다.
    * 결국 다시 linear classifier 가 된다.
  * 종류
    * Sigmoid : 전통적으로 많이 쓰임
      * $\sigma (x) = \frac{1}{1 + e^{-x}}$
    * tanh
      * $\tanh (x)$
    * **ReLU : Rectified Linear Unit** (good default choice for most problems)
      * $max(0, x)$
    * Leaky ReLU
      * $max(0.1x, x)$
    * Maxout
      * $max(w_1^T x + b_1, w_2^T x + b_2)$
    * ELU
      * $x$ if $x \geq 0$
      * $\alpha (e^x - 1)$ if $x < 0$

## Why Neural Network?
* Biological neuron vs. Artificial neuron
  * Biological neuron 이 좀 더 복잡한 구조고, Artificial neuron 는 위계가 있다.
  * Artificial neuron 이 Biological neuron 의 구조를 따라하기도 어렵고 그렇게 할 필요도 없다.
* `Space Warping` : A geometric interpretation of nonlinear activation
  * 선형변환
    * $h = Wx$ ($x, h \in R^2$)
    * $h1 = w_1^T x$, $h2 = w_2^T x$
    * x 공간에서 Linear하게 구분지을 수 없는 점들은 h 공간으로 Feature transform 을 해도 선으로 나누기 힘들다.
  * 비선형변환
    * $h = ReLU(Wx) = max(0, Wx)$
    * h 공간에서 linear 하게 선을 긋는다고 하더라도 x 공간 (original space) 에서는 non-linear 한 decision boundary 를 긋게 된다.
    * 레이어를 쌓을수록 꺾임이 많아진다.
      * More hidden units = more capacity
      * hidden unit 이 많아질수록 overfitting 가능성이 큰데, 보통 이 때 hidden unit 수를 줄이기보다 regularization을 더 세게 한다.

## A theoretical guarantee from neural networks
* Universal approximation
  * $f(x;W^*) \approx f$
  * 임의의 $f : R^N \rightarrow R^M$ 과 임의의 precision $\epsilon > 0$ (0.01, 0.001 등) 에 대하여, ${max}_{x \in 정의역} \mid f(x) - f(x;W^*) \mid < \epsilon$ 이 되는 $W^*$ 가 존재한다.
  * neural network가 weight를 통해 임의의 다양한 곡선을 표현할 수 있다.
    * 4K 개의 hidden units 을 이용하여 K개의 bump function 을 만들 수 있다.
    * 여러 개의 bump function을 이용하여 $f$ 에 근사할 수 있다.
  * 문제의 해가 존재한다. but 어떻게 구할건지에 대한 답은 없다.

## Neural networks are seldomly convex
* neural network는 convex (볼록) 하지 않다...
* Convex Functions
  * $f : X \subseteq R^N \rightarrow R$ is **convex** if
    * for all $x_1, x_2 \in X$, $t \in [0, 1]$,
    * $f(tx_1 + (1 - t)x_2) \leq tf(x_1) + (1 - t)f(x_2)$
    * 볼록인 곡선 위에 임의의 선을 그었을 때 늘 선분이 곡선의 위에 있다는 성질을 묘사한 것
  * 장점
    * 최솟값을 찾기 쉽다. (easy to optimize)
    * global minimum 으로 수렴하는 것을 보장하는 알고리즘이 있다. (Gradient Descent, Newton's algorithm)
* Linear classifiers optimize a convex function
* **Most neural networks need nonconvex optimization**
  * $L(W)$ 를 $W$ 에 대해서 최소화시키는 문제
  * global minimum 에 수렴한다는 보장이 거의 없음
  * global minimum 아닌 곳에 수렴해서 그걸 사용하더라도 경험적으로 잘 동작함


# Self-check
## 문제1
* 뉴럴 네트워크 기반의 이미지 분류 방법은 '피처 추출 뒤 분류기 학습' 방식의 전통적 이미지 분류 방법과 비교하여 어떤 점이 다른가요? - 2-3 lines 
* 답 : '피처 추출 뒤 분류기 학습' 방식에서 피처 추출을 하는 과정에서 사람의 직관이 들어간다. 반면 Neural Networks 는 피처 추출을 하는 방법까지 데이터로부터 배운다.

## 문제2
* relu activation에 기반한 5-hidden-layer network (output node s까지 포함하면 6-layer-network)를 설명하세요. (lec3의 1번 답안에서 어느 부분이 변경되는지부터 시작하세요. 또한 답안에는 이 그림도 그려져 있어야 하며, 그림에 등장하는 벡터/행렬의 size 또한 적절하게 정의되어 있어야 합니다.) - 0.5 pages
![image](https://user-images.githubusercontent.com/35680202/114651334-bb300c00-9d1e-11eb-8356-d9fd46209dd5.png)
* 답
  * training set : $\{(x_i,y_i)\}^{N}_{i=1}$
    * $X : (N, D)$ (여기서, $D = 3072$)
    * $y : (N, )$
      * $y_i \in \{0, 1, \dotsc, C-1\}$ (여기서, $C = 10$)
  * 선형변환된 score -> 선형+비선형이 반복되어 변환된 score
    * $W_1 : (D, M_1)$, $b_1 : (M_1, )$
    * $H_1 = max(0, X W_1 + b_1) : (N, M_1)$
    * $W_2 : (M_1, M_2)$, $b_2 : (M_2, )$
    * $H_2 = max(0, H_1 W_2 + b_2) : (N, M_2)$
    * $W_3 : (M_2, M_3)$, $b_3 : (M_3, )$
    * $H_3 = max(0, H_2 W_3 + b_3) : (N, M_3)$
    * $W_4 : (M_3, M_4)$, $b_4 : (M_4, )$
    * $H_4 = max(0, H_3 W_4 + b_4) : (N, M_4)$
    * $W_5 : (M_4, M_5)$, $b_5 : (M_5, )$
    * $H_5 = max(0, H_4 W_5 + b_5) : (N, M_5)$
    * $W_6 : (M_5, C)$, $b_6 : (C, )$
    * $S = H_5 W_6 + b_6 : (N, C)$ 

## 문제3
* neural network에서 nonlinear activation의 역할은 무엇입니까? space warping을 예로 들어 설명하세요. - 0.5 pages 
* 답 : x 공간에서 Linear하게 구분지을 수 없는 점들은 선형변환하여 h 공간으로 Feature transform 을 해도 선으로 나누기 힘들다. 선형변환을 여러 번 한다고 하더라도 한번의 선형변환을 한거나 마찬가지인 효과밖에 낼 수 없다. 반면에 space warping 을 통해서, nonlinear activation function으로 비선형변환을 한 h 공간에서는, linear 하게 선을 그어도 원래의 x 공간 (original space) 에서는 non-linear 한 decision boundary 가 그어지는 효과를 가져올 수 있다. 따라서 비선형 함수는 선형 뿐만 아니라 더 자유도가 높은 decision boundary 를 가진 classifier 를 만들 수 있는 역할을 한다.

## 문제4
* convex function의 정의는 무엇입니까? 어떤 함수가 convex인지 확인하는 방법은 무엇입니까? convex function은 어떤 장점이 있나요? neural network 기반의 loss function은 convex function인가요? - 0.5 pages
* 답 : 
  * Convex Functions 정의
    * $f : X \subseteq R^N \rightarrow R$ is **convex** if
      * for all $x_1, x_2 \in X$, $t \in [0, 1]$,
      * $f(tx_1 + (1 - t)x_2) \leq tf(x_1) + (1 - t)f(x_2)$
    * 볼록인 곡선 위에 임의의 선을 그었을 때 늘 선분이 곡선의 위에 있다.
  * 확인하는 방법
    * f(x)의 2차미분값(2nd derivatives)이 0보다 크거나 같으면 convex 하다.
  * 장점
    * 최솟값을 찾기 쉽다. (easy to optimize)
    * global minimum 으로 수렴하는 것을 보장하는 알고리즘이 있다. (GD, Newton’s)
  * neural network 기반의 loss function 은  현실에서 대부분 convex function 이 아니다. 따라서 global minimum 에 수렴한다는 보장이 거의 없지만, global minimum 아닌 곳에 수렴해서 그걸 사용하더라도 경험적으로 잘 동작한다고 한다.

## 문제5
* 실습 4에서는 CIFAR-10 데이터셋을 대상으로 Two-layer neural network의 parameter를 훈련하고 hyperparameter를 튜닝하였습니다. 훈련시에 최적화 기법으로는 SGD를 사용하였죠.
  * (a) learnable parameter의 개수가 몇개인지 유도하세요. Input data의 size는 D=3072 (32x32x3, bias 트릭 쓰지 않았음), 중간의 hidden layer의 size는 H=20, class 개수는 C=10을 가정하고, bias term의 존재를 가정하세요. (힌트: f(x;W) 에서 W에 대응하는 객체들이 누구입니까?)
    * 답
      * $X : (50k, 3072)$
      * $W_1 : (3072, 20)$, $b_1 : (20,)$
      * $W_2 : (20, 10)$, $b_2 : (10,)$
      * 따라서, $(3072 \times 20 + 20) + (20 \times 10 + 10) = 61,460 + 210 = 61,670$
  * (b) hyperparameter로 간주할 수 있는 것들은 무엇입니까? 최소 3 객체 이상 서술하세요.
    * 답
      * regularization 변수 : Scalar giving regularization strength.
      * learning rate : Scalar giving learning rate for optimization.
      * number of iterations : Number of steps to take when optimizing.
      * batch size : Number of training examples to use per step.