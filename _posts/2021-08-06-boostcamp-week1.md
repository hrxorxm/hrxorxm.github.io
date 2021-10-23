---
layout: post
title:  "[Boostcamp AI Tech] 1주차 - Python & AI Math"
subtitle:   "Naver Boostcamp AI Tech Level1 U Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level1-U-Stage]
use_math: true
---

# 부스트캠프 1주차

## [1강] 벡터

* 성분곱(Hadamard product) : 같은 모양을 가지는 벡터끼리의 곱
* 노름(norm) : 원점에서부터의 거리
  * $L_1$​-노름 : 각 성분의 변화량의 절대값을 모두 더함
    * 예 : Robust 학습, Lasso 회귀
  * $L_2$​​-노름 : 유클리드 거리 계산 `np.linalg.norm`
    * 예 : Laplace 근사, Ridge 회귀
  * 노름의 종류에 따라 기하학적 성질이 달라진다.
  * 두 벡터 사이의 거리 : 벡터의 뺄셈, 노름 이용
  * 두 벡터 사이의 각도 : 내적(inner product), $L_2$​-노름 이용
* 내적(inner product) `np.inner`
  * $<x, y> = \parallel x \parallel_2 \parallel y \parallel_2 \cos \theta$​​ : 정사영(orthogonal projection)된 벡터 $Proj(x)$​의 길이를 $\parallel y \parallel$ 만큼 조정한 값
  * 두 벡터의 유사도(similarity)를 측정하는데 사용 가능

## [2강] 행렬

* 행렬(matrix) : 벡터를 원소로 가지는 2차원 배열
* 전치행렬(transpose matrix) : 행과 열의 인덱스가 바뀐 행렬 `x.T`
* 행렬의 덧셈, 뺄셈, 성분곱, 스칼라곱은 벡터와 차이가 없다.
* 행렬 곱셈(matrix multiplication) : i번째 행벡터와 j번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 계산
  * `np.inner`는 i번째 행벡터와 j번째 행벡터 사이의 내적을 구함
* 역행렬(inverse matrix) : 행과 열의 숫자가 같고 행렬식(determinant)이 0이 아닌 경우 구할 수 있다. `np.linalg.inv(x)`
  * $A A^{-1} = A^{-1} A = I$
* 유사역행렬(pseudo-inverse) 또는 무어-펜로즈(Moore-Penrose) 역행렬 `np.linalg.pinv(x)`
  * $n \geq m$​ 인 경우 $A^{+} = (A^{T} A)^{-1} A^{T}$, $A^{+}A = I$​
  * $n \leq m$ 인 경우 $A^{+} = A^{T} (A A^{T})^{-1}$, $AA^{+} = I$
* 연립방정식 풀기 ($n \leq m$ 인 경우)
  * $Ax = b \Rightarrow x = A^{+}b = A^{T} (AA^{T})^{-1}b$​
  * 무어-펜로즈 역행렬을 이용하면 해를 하나 구할 수 있다.
* 선형회귀분석 ($n \geq m$ 인 경우)
  * $X \beta = \hat{y} \approx y \Rightarrow \beta = X^{+}y = (X^{T}X)^{-1}X^{T}y$​​​   ($\underset{\beta}{min} \parallel y - \hat{y} \parallel_2$, 즉, $L_2$​-노름을 최소화)
  * 선형회귀분석은 연립방정식과 달리 행이 더 크므로 방정식을 푸는건 불가능하고, $y$에 근접하는 $\hat{y}$​​를 찾을 수 있다. (+추가 : y절편(intercept) 항을 직접 추가해야 한다.)

## [3강] 경사하강법

* 용어
  * 미분(differentiation) : 변화율의 극한(limit) `sympy.diff`
  * 경사상승법(gradient ascent) : 미분값을 더함, 함수의 극댓값의 위치를 구할 때 사용
  * 경사하강법(gradient descent) : 미분값을 뺌, 함수의 극소값의 위치를 구할 때 사용
  * 편미분(partial differentiation) : 벡터가 입력인 다변수 함수일 때의 미분
  * 그레디언트(gradient) 벡터 : $\nabla f = (\partial_{x_1}f, \partial_{x_2}f, ..., \partial_{x_d}f)$
* 목표 : 선형회귀의 목적식을 최소화하는 $\beta$를 찾아야 한다.
  * 목적식이 $\parallel y - X \beta \parallel_2$​ 일 때
    * 그레디언트 벡터 식
      * $\nabla_{\beta}\parallel y - X \beta \parallel_2 = (\partial_{\beta_1} \parallel y - X \beta \parallel_2, ..., \partial_{\beta_d} \parallel y - X \beta \parallel_2)$​ 이고,
      * $\partial_{\beta_k} \parallel y - X \beta \parallel_2 = \partial_{\beta_k}  \{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \sum_{j=1}^{d} X_{ij} \beta_{j})^2 \}^{1/2} = - \frac{X^T_k (y - X \beta)}{n \parallel y - X \beta \parallel_2}$​ 이므로,​​
      * $\nabla_{\beta}\parallel y - X \beta \parallel_2 = - \frac{X^T (y - X \beta)}{n \parallel y - X \beta \parallel_2}$​​
      * 복잡한 계산이지만 사실 $X \beta$​를 계수 $\beta$​에 대해 미분한 결과인 $X^T$​​만 곱해지는 것
    * 경사하강법 알고리즘 : $\beta^{(t+1)} \leftarrow \beta^{(t)} - \lambda \nabla_{\beta}\parallel y - X \beta^{(t)} \parallel$
  * 목적식이 $\parallel y - X \beta \parallel_2^2$​ 일 때
    * 그레디언트 벡터 식 : $\nabla_{\beta}\parallel y - X \beta \parallel_2^2 = - \frac{2}{n} X^T (y - X \beta)$
    * 경사하강법 알고리즘 : $\beta^{(t+1)} \leftarrow \beta^{(t)} + \frac{2 \lambda}{n} X^T (y - X \beta^{(t)})$​
* 특징
  * 학습률(lr)과 학습횟수(epoch)가 중요한 하이퍼파라미터(hyperparameter)
  * convex한 함수에 대해서는 수렴 보장

## [4강] 확률적 경사하강법

* 특징
  * 일부 데이터를 활용하여 업데이트
  * 연산자원을 효율적으로 활용 가능 (메모리 등 하드웨어 한계 극복 가능)
  * **non-convex 목적식을 최적화할 수 있다. (머신러닝 학습에 더 효율적)**
* 원리 : 미니배치 연산
  * **목적식이 미니배치마다 조금씩 달라진다.**
  * 따라서 극소점/극대점이 바뀔 수 있다. (극소점에서 탈출할 수 있다.)

## [5강] 딥러닝 학습방법

* 소프트맥스(softmax) 함수
  * 모델의 출력을 확률로 해석할 수 있게 변환해주는 연산
  * 학습할 때 사용되고, 보통 추론하는 경우에는 사용하지 않는다.

* 활성함수(activation function)
  * 비선형함수(nonlinear function)
  * 활성함수가 없으면 선형모형과 차이가 없다.
  * 소프트맥스 함수와는 달리, 계산할 때 그 자리의 실수값만 고려한다.
  * 종류 : sigmoid, tanh, ReLU 등

* 신경망 : 선형모델과 활성함수를 합성한 함수
  * 다층퍼셉트론(MLP) : 신경망이 여러층 합성된 함수
  * universal approximation theorem : 이론적으로 2층 신경망으로 임의의 연속함수를 근사할 수 있다.
  * 층이 깊으면 더 적은 노드로 목적함수 근사가 가능하다.

* 순전파(forward propagation) : 레이어의 순차적인 신경망 계산
* 역전파(backpropagation) : 출력에서부터 역순으로 계산
  * 원리 : 연쇄법칙(chain-rule) 기반 자동미분(auto-differentiation)
  * 특징 : 순전파 결과와 미분값을 둘 다 저장해야 한다.

## [6강] 확률론

* 딥러닝은 확률론 기반의 기계학습 이론 바탕 (손실함수 작동원리)
  * L2-노름 : 예측오차의 분산을 가장 최소화하는 방향으로 학습
  * 교차엔트로피(cross-entropy) : 모델 예측의 불확실성을 최소화하는 방향으로 학습

* 확률변수 : 관측가능한 데이터, 함수로 해석
  * 확률변수 구분 : (데이터 공간이 아니라) 확률분포에 의해 결정
  * 이산확률변수(discrete) : 확률변수가 가질 수 있는 모든 경우의 수의 확률을 더해서 모델링
    * 확률질량함수 : $P(X \in A) = \sum_{x \in A} P(X=x)$
  * 연속확률변수(continuous) : 데이터 공간에 정의된 확률변수의 밀도(density) 위에서 적분을 통해 누적확률분포의 변화율을 모델링
    * 밀도함수 :  $P(X \in A) = \int_A P(x) dx$

* 확률분포 : 데이터를 표현하는 초상화, 기계학습을 통해 확률분포 추론
  * 결합확률분포 $P(x,y)$
    * 주어진 데이터의 결합분포 $P(x,y)$를 이용하여 원래 확률분포 $D$ 모델링
  * 주변확률분포 $P(x)$ : 입력 $x$에 대한 정보
    * $P(x) = \sum_y P(x, y)$ or $P(x) = \int_y P(x,y) dy$
  * 조건확률분포 $P(x \| y)$​ : 특정 클래스일 때의 데이터의 확률분포
    * 데이터 공간에서 입력 $x$와 출력 $y$​ 사이의 관계 모델링
  * 조건부확률 $P(y \| x)$ : 입력변수 $x$​에 대해 정답이 $y$​​​일 확률(분류 문제)
    * 선형모델과 소프트맥스 함수의 결합 등 
  * 조건부기대값 $E[y \| x]$ : 입력변수 $x$에 대해 정답이 $y$​일 밀도(회귀 문제)
    * $E_{y \sim P(y \| x)}[y \| x] = \int_y y P(y \| x) dy$
    * $E\parallel y - f(x)\parallel_2$ ($L_2$-노름)을 최소화하는 함수 $f(x)$와 일치 (수학적으로 증명됨)

* 통계적 범함수(statistical functional) : 확률분포에서 데이터를 분석하는데 사용
  * 기대값(expectation) : 데이터를 대표하는 통계량, 평균(mean)
  * 분산(variance), 첨도(skewness), 공분산(covariance) 등 : 기대값을 통해 계산 가능

* 몬테카를로(Monte Carlo) 샘플링 : 확률분포를 명시적으로 모를 때, 데이터를 이용하여 기대값 계산
  * 독립추출이 보장되면, 대수의 법칙(law of large number)에 의해 수렴성 보장 = 수를 많이 뽑으면 정답에 가까워질 것이라는 뜻
  * 부정적분 방식으로는 어려웠던 적분을 계산할 수 있다.
  * 적절한 샘플사이즈로 계산해야 오차범위를 줄일 수 있다.
  * 특징 : 가능한 모든 수를 시도하는 것이 전제로 들어감, 약간 확률적인 완전탐색/브루트포스 같은 느낌

## [7강] 통계학

* 통계적 모델링 : (적절한 가정 위에서 근사적으로) 확률분포 추정(inference)
  * **모수적(parametric) 방법론** : 데이터가 특정 확률 분포를 따른다고 선험적으로(a priori) 가정한 후 그 분포를 결정하는 모수(parameter)를 추정하는 방법
  * **비모수(nonparametric) 방법론** : 특정 확률분포를 가정하지 않고 **데이터에 따라** 모델의 구조 및 모수의 개수가 유연하게 **바뀌는 것**
    * 기계학습의 많은 방법이 비모수 방법론
  
* **최대가능도 추정법(maximum likelihoof estimation, MLE)** : 이론적으로 가장 가능성이 높은 모수를 추정하는 방법
  * 가능도(likelihood) 함수 : 모수 $\theta$를 따르는 분포가 $x$를 관찰할 가능성(확률X)
    * $\hat{\theta}_{MLE} = \underset{\theta}{argmax} L(\theta ; x) = \underset{\theta}{argmax} P(x \| \theta)$​
  * 데이터 집합 $X$가 독립적으로 추출되었을 경우 **로그가능도**를 최적화
    * $L(\theta; X) = \Pi_{i=1}^{n} P(x_i \| \theta) \Rightarrow \log L(\theta; X) = \sum_{i=1}^{n} \log P(x_i \| \theta)$
    * 경사하강법으로 최적화할 때 미분 연산의 연산량을 $O(n^2)$에서 $O(n)$으로 줄일 수 있다.
    * 경사하강법의 경우 음의 로그가능도(negative log-likelihood)를 최적화
  * 딥러닝에서 최대가능도 추정법 예시
    * 분류문제에서 소프트맥스 벡터는 카테고리분포의 모수 ($p_1, ..., p_K$)를 모델링
    * 원핫벡터로 표현한 정답레이블 $y = (y_1, ..., y_K)$​을 관찰데이터로 이용해 확률분포인 소프트맥스 벡터의 로그가능도를 최적화
    * $\hat{\theta_{MLE}} = \underset{\theta}{argmax} \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log (MLP_\theta (x_i)_k)$
  * MLE로 추정하는 모델학습방법론은 확률분포의 거리를 최적화하는 것과 밀접한 관련이 있다.
    * **쿨백-라이블러 발산(Kullback-Leibler Divergence, KL)** : 두 확률분포가 어느 정도 닮았는지를 나타내는 척도
      * 이산확률변수일 떄 : $KL(P \parallel Q) = \sum_{x \in X} P(x) \log (\frac{P(x)}{Q(x)})$
      * 연속확률변수일 때 : $KL(P \parallel Q) = \int_{X} P(x) \log (\frac{P(x)}{Q(x)}) dx$
      * 분해 : $KL(P \parallel Q) = - E_{x \sim P(x)} [\log Q(x)] + E_{x \sim P(x)} [\log P(x)]$
      * Q. 쿨백-라이블러 발산이 항상 0보다 크거나 같은 이유? 어떻게 $KL(P \parallel Q) = \int_{X} P(x) \log (\frac{P(x)}{Q(x)}) dx$ 이 식이 항상 $KL(P \parallel Q) \geq 0$ 을 만족할 수 있는 것인지 궁금
        * [증명](https://en.wikipedia.org/wiki/Gibbs%27_inequality#Proof) : **The information entropy of a distribution P** is less than or equal to **its cross entropy with any other distribution Q**
          ![image](https://user-images.githubusercontent.com/35680202/128598212-aae71591-9b5d-4d31-9435-27cf4a8d0756.png)
    * 분류문제에서 정답레이블을 $P$​, 모델 예측을 $Q$​​라 두면 최대가능도 추정법은 쿨백-라이블러 발산을 최소화하는 것과 같다. 

## [8강] 베이즈 통계학

* 조건부 확률 $P(A \| B) = \frac{P(A \cap B)}{P(B)}$​ : 사건 $B$​가 일어난 상황에서 사건 $A$​​가 발생할 확률
  * $P(B \| A) = \frac{P(A \cap B)}{P(A)} = P(B) \frac{P(A\|B)}{P(A)}$​
  * $P(B \| A) = P(B)$이고, $P(A \| B) = P(A)$일 때, 두 사건 $A, B$는 독립

* 베이즈 정리 : 데이터가 새로 추가될 때 조건부 확률을 이용하여 정보를 갱신하는 방법
  * $P(\theta \| D) = P(\theta) \frac{P(D \| \theta)}{P(D)}$
    * $P(\theta \| D)$​​ : 사후확률(posterior), 데이터를 관찰했을 때 이 모수나 가설(hypothesis)이 성립할 확률
    * $P(\theta), P(\neg \theta)$ : 사전확률(prior), 모델링하기 이전에 모수나 가설에 대해 주어진 확률
    * $P(D \| \theta)$​​ : 가능도(likelihood) : 현재 주어진 모수, 가정에서 이 데이터가 관찰될 확률
    * $P(D)$ : Evidence, 데이터 자체의 분포
  * 가능도와 Evidence를 통해 사전확률을 사후확률로 업데이트 한다.
  * 새로운 데이터가 들어왔을 때, 앞서 계산한 사후확률을 사전확률로 사용하여 사후확률을 갱신할 수 있다.

* 용어
  * True Positive = $P(D \| \theta) P(\theta)$
  * False Positive (1종 오류) = $P(D \| \neg \theta) P(\neg \theta)$
  * False Negative (2종 오류) = $P(\neg D \| \theta) P(\theta)$
  * True Negative = $P(\neg D \| \neg \theta) P(\neg \theta)$
  * 정밀도(Precision) = $P(\theta \| D)$​​ = TP / (TP + FP)
  * 민감도(Recall) = $P(D \| \theta)$
  * 오탐(False alarm) = $P(D \| \neg \theta)$
  * 특이도(specificity) = $P(\neg D \| \neg \theta)$​

* 인과관계
  * 조건부 확률만 가지고 인과관계(causality)를 추론하는 것은 불가능
  * 조정(intervention) 효과를 통해 중간 개입을 제거할 수 있다.

## [9강] CNN

* 커널(kernel)을 입력벡터 상에서 움직이며 선형모델과 합성함수가 적용되는 구조
* Convolution 연산의 수학적인 의미
  * $*$ : convolution 연산을 뜻함
  * continuous : $[f * g] (x) = \int_{R^d} f(z)g(x+z)dz = \int_{R^d} f(x+z)g(z)dz = [g * f] (x)$​
  * discrete : $[f * g] (i) = \sum_{a \in Z^d} f(a)g(i+a) = \sum_{a \in Z^d} f(i + a)g(a) = [g * f] (i)$
  * 커널을 이용해 신호(signal)를 국소적(local)으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것
  * 커널은 정의역 내에서 움직여도 변하지 않음(translation invariant)
  * 원래 신호처리에서 convolution은 transpose를 먼저 취한다. 따라서 CNN에서 사용하는 연산은 엄밀히 말하면 cross-correlation 연산이다.
* 다양한 차원에서 계산 가능 ($f$는 커널, $g$는 입력)
  * 1D-conv : $[f * g] (i) = \sum_{p=1}^{d} f(p)g(i+p)$
  * 2D-conv : $[f * g] (i, j) = \sum_{p, q} f(p, q)g(i+p, j+q)$
  * 3D-conv : $[f * g] (i, j, k) = \sum_{p, q, r} f(p, q, r)g(i+p, j+q, k+r)$
* Convolution 연산의 역전파 : 역전파 계산시에도 convolution 연산
  * $\frac{\partial}{\partial x} [f * g] (x) = \frac{\partial}{\partial x} \int_{R^d} f(y) g(x-y)dy = \int_{R^d} f(y) \frac{\partial g}{\partial x} (x-y)dy = [f * g'] (x)$
  * forward pass : $O_i = \sum_{j} w_j x_{i+j-1}$
  * backward pass : $\frac{\partial L}{\partial W_i} = \sum_{j} \delta_j x_{i+j-1}, \frac{\partial L}{\partial x_i} = \sum_{j} \delta_j w_{i-j+1}$​​
    ![image](https://user-images.githubusercontent.com/35680202/128604387-2be5b830-65d3-4c86-bb14-07a2ff97477c.png)

## [10강] RNN

* 시퀀스(sequence) 데이터 : 소리, 문자열, 주가 등
  * 시계열(time-series) 데이터 : 시간 순서에 따라 나열된 데이터
  * 독립동등분포(independent and identical distribution) 가정을 위배하기 쉽다.
    * 개가 사람을 물었다 != 사람이 개를 물었다. : 위치가 바뀜으로써 의미도 달라진다.
    * 과거의 정보로 미래를 예측할 때, 과거 정보에 손실이 있으면 데이터의 확률분포도 바뀌게 된다. (반드시 아주 먼 과거 정보까지 가질 필요는 없지만)

* 시퀀스 데이터를 다룰 수 있는 모델
  1. 조건부확률 이용(베이즈 법칙)
    * $P(X_1, ..., X_t) = P(X_t \| X_1, ..., X_{t-1}) P(X_1, ..., X_{t-1})$
    * $X_t \sim P(X_t \| X_{t-1}, ..., X_1)$​
    * $X_{t+1} \sim P(X_{t+1} \| X_t, X_{t-1}, ..., X_1)$
  2. $AR(\tau)$ (Autoregressive Model) : 자기회귀모델, 고정된 길이 $\tau$​ 만큼의 시퀀스만 사용
  3. 잠재 AR 모델 : 바로 이전 정보 $X_{t-1}$​​를 제외한 나머지 정보들 $X_{t-2}, ..., X_1$​​을 $H_t$​​​라는 잠재변수로 인코딩에서 활용하는 방법
  4. RNN 모델 : 잠재변수 $H_t$​​를 신경망을 통해 반복해서 사용하여 시퀀스 데이터의 패턴을 학습하는 모델
    * $X_t \sim P(X_t \| X_{t-1}, H_t)$​, $H_t = Net_{\theta} (H_{t-1}, X_{t-1})$​
    * $X_{t+1} \sim P(X_{t+1} \| X_t, H_{t+1})$​

* forward
 * MLP
   * $H_t = \sigma (X_t W^{(1)} + b^{(1)})$
   * $O_t = H_t W^{(2)} + b^{(2)}$
 * RNN : MLP와 유사한 모양이다.
   * $H_t = \sigma (X_t W_X^{(1)} + H_{t-1} W_H^{(1)} + b^{(1)})$​ ($H_{t-1} W^{(1)}_H$​ term 추가)
   * $O_t = H_t W^{(2)} + b^{(2)}$​
   * $H_{t+1} = \sigma (X_{t+1} W_X^{(1)} + H_{t} W_H^{(1)} + b^{(1)})$​
   * $O_{t+1} = H_{t+1} W^{(2)} + b^{(2)}$​

* backward
  * BPTT(Backpropagation Through Time) : RNN의 역전파 방법
    * 잠재변수의 연결그래프에 따라 순차적으로 계산
    * 미분의 곱으로 이루어진 항 계산 : 시퀀스 길이가 길어질수록 이 항은 불안정해지기 쉽다. (vanishing gradient)
  * truncated BPTT : 시퀀스 길이가 길어지는 경우 역전파 알고리즘 계산이 불안정해지므로 길이를 끊는다.

* LSTM, GPU : 긴 시퀀스를 처리하기 위해 등장한 네트워크

## [Python] numpy

* ndarray 객체
  * C의 Array를 사용하여 배열 생성 (dynamic typing을 지원하지 않음)
  * properties
    * dtype : 데이터의 타입
    * shape : dimension 구성
      * rank 0 : scalar / rank 1 : vector / rank 2 : matrix / rank n : n-tensor
    * ndim : number of dimensions (rank의 개수)
    * size : data의 개수 (element의 개수)
    * nbytes : 용량

* Handling shape
  * reshape : shape의 크기를 변경, element의 갯수(size)와 순서는 동일
    * -1 : size를 기반으로 개수 선정
    ```python
    a = np.array([5, 6])
    a.reshape(-1, 2) # array([[5, 6]])
    a[np.newaxis, :] # array([[5, 6]])
    ```
  * flatten : 1차원으로 변환

* indexing & slicing
  * `a[0][0]` 또는 `a[0, 0]`
  * `a[1, :2]` 와 `a[1:2, :2]` 의 dimension, shape이 달라진다.
    ```python
    a = np.array([[1, 2, 5, 8],
                  	  [1, 2, 5, 8],
                 	  [1, 2, 5, 8],
                 	  [1, 2, 5, 8]])
    a[1:2, :2].shape # (1, 2)
    a[1, :2].shape # (2,)
    ```

* creation function
  * arange
    * `np.arange(끝)`
    * `np.arange(시작, 끝, step)`
  * ones, zeros and empty
    * `np.zeros(shape, dtype, order)` : shape은 튜플값으로 넣기
    * `np.empty` : shape만 주어지고 빈 ndarray 생성 (메모리 초기화 안됨)
    * `np.ones_like(test_matrix)` : 기존 ndarray의 shape 크기만큼의 ndarray 반환
  * identity : 단위행렬 생성
    * `np.identity(n)` : n은 number of rows
  * eye : 대각선이 1인 행렬
    * `np.eye(N=3, M=5, k=2)` : 시작 인덱스를 k로 변경할 수 있다.
  * diag : 대각 행렬 값을 추출
    * `np.diag(matrix)`
  * random sampling : 각 분포의 모수와 size를 인자로 넣는다.
    * `np.random.uniform(low, high, size)` : 균등 분포
    * `np.random.normal(loc=mean, scale=std, size)` : 정규 분포

* operation functions
  * axis : 모든 operation function을 실행할 때 기준이 되는 dimension 축
  * sum, mean, std 외에도 지수함수, 삼각함수, 하이퍼볼릭 함수 등 수학 연산자 제공
  * concatenate : numpy array를 붙이는 함수
    ```python
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    np.vstack((a, b)) # array([[1, 2, 3], [2, 3, 4]])
    np.hstack((a, b)) # array([1, 2, 3, 2, 3, 4])
    
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    np.vstack((a, b)) # array([[1], [2], [3], [4], [5], [6]])
    np.hstack((a, b)) # array([[1, 2], [2, 3], [3, 4]])
    ```
    * `np.concatenate((a, b), axis)` 

* array operation
  * element-wise operations : shape이 같은 배열 간 연산, 기본적인 사칙연산 지원
  * broadcasting : shape이 다른 배열 간 연산 지원 (주의 필요)
  * dot product : Matrix의 기본 연산
  * transpose : 전치행렬 반환

* comparisons
  * 배열의 크기가 동일할 때, element간 비교(`>, ==, <`)의 결과를 Boolean type으로 반환
  * `np.all(a < 10)`, `np.any(a < 10)` : 각각 and, or 조건에 따른 boolean 값 반환
  * `np.logical_and()`, `np.logical_not()`, `np.logical_or()`
  * `np.isnan(a)`, `np.isfinite(a)`
  * `np.where(condition)` : index 값 반환
    ```python
    a = np.array([1, 2, 3, 4, 5])
    np.where(a > 2) # (array([2, 3, 4]),)
    ```
  * `np.where(condition, TRUE, FALSE)` : condition에 따라 True일 때의 값과 False일 때의 값을 넣을 수 있다.
    ```python
    a = np.array([1, 2, 3, 4, 5])
    np.where(a > 2, 2, a) # array([1, 2, 2, 2, 2])
    ```
  * `np.argmax(a, axis)`, `np.argmin(a, axis)` : 최대값 또는 최소값의 index 반환
  * `np.argsort()` : 값을 정렬한 인덱스 값을 반환

* boolean & fancy index
  * boolean index
    ```python
    condition = a > 3
    a[condition]
    ```
  * fancy index
    ```python
    a = np.array([2, 4, 6, 8])
    b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer
    a[b] # array([2, 2, 4, 8, 6, 4])
    a.take(b) # 위와 동일
    
    a = np.array([[1, 4], [9, 16]])
    b = np.array([0, 0, 1, 1, 0], int)
    c = np.array([0, 1, 1, 1, 1], int)
    a[b, c] # array([1, 4, 16, 16, 4])
    ```
* numpy data i/o
  * `np.loadtxt()` & `np.savetxt()` : text type의 데이터
  * `np.load()` & `np.save()` : npy(numpy object)로 저장 및 로드

## [Python] pandas

* 특징
  * numpy와 통합하여 사용하기 쉬워짐
  * Tabular 데이터 처리하는 데에 사용

* 용어
  * Series : DataFrame 중 하나의 Column에 해당하는 데이터의 모음 Object
    * numpy.ndarray의 subclass
    * index, values 를 가지고 있음
  * DataFrame : Data Table 전체를 포함하는 Object
    * numpy array-like
    * index, columns 를 가지고 있음
    * 각 컬럼은 다른 데이터 타입 가능, 컬럼 삽입/삭제 가능

* 기능
  * indexing : `loc`는 index 이름, `iloc`은 index number
  * `lambda`, `map`, `replace`, `apply`, `applymap` 등의 함수 사용 가능
  * pandas built-in functions : `describe`, `unique`, `sum`, `isnull`, `sort_values`, `corr`, `cor`, `corrwith`, `value_counts` 등
  * Groupby : 그룹별로 모아진 DataFrame 반환
    * `df.groupby(["Team", "Year"])["Points"].sum()` : 두 개의 컬럼으로 groupby 할 경우, index가 두 개 생성(Hierarchical index)
    * grouped : Aggregation(`agg`), Transformation(`transform`), Filtration(`filter`) 가능
      ```python
      grouped = df.groupby("Team") # generator 형태 반환
      for name, group in grouped: # Tuple 형태로 그룹의 key, value 값이 추출됨
          print(name) # key : Team name
          print(group) # value : DataFrame 형태
      ```
  * Pivot Table : index 축은 groupby와 동일, column에 라벨링 값 추가 (엑셀과 비슷)
    ```python
    df.pivot_table(values=["duration"],
                      index=[df.month, df.item], # index
                      columns=df.network, # columns
                      aggfunc="sum",
                      fill_value=0)
    ```
  * Crosstab : 주로 네트워크 형태의 데이터 다룰 때
    ```python
    pd.crosstab(index=df_movie.critic, 
                    columns=df_movie.title, 
                    values=df_movie.rating,
                    aggfunc="first").fillna(0)
    ```
  * Merge : SQL에서 많이 사용하는 Merge와 같은 기능
    * INNER JOIN / LEFT JOIN / RIGHT JOIN / FULL JOIN
  * Concat : 같은 형태의 데이터를 붙이는 연산작업
  * Persistence
    * Database connection
    * XLS persistence : openpyxls 또는 XlsxWrite 사용
    * pickle 파일 형태로 저장

## 추가 정리

### 텍스트 다루기
* [파이썬 문자열 메서드 문서](https://docs.python.org/ko/3/library/stdtypes.html#string-methods)
  * `capitalize()` : 첫 문자가 대문자이고 나머지가 소문자인 문자열의 복사본을 돌려줍니다.
  * `isdigit()` : 문자열 내의 모든 문자가 디짓이고, 적어도 하나의 문자가 존재하는 경우 True를 돌려주고, 그렇지 않으면 False를 돌려줍니다.
* [정규표현식 re 라이브러리 문서](https://docs.python.org/ko/3/library/re.html#regular-expression-objects)

### 정규표현식 사용법

* 메소드 (참고 : [정규표현식 re 라이브러리 문서](https://docs.python.org/ko/3/library/re.html#module-contents))
  * `re.search(pattern, string)` : string 전체를 검색하여 정규식 pattern과 일치하는 첫번째 위치를 찾는다. (match object 또는 None 반환)
  * `re.sub(pattern, repl, string)` : string에서 pattern과 일치하는 곳을 repl로 치환하여 얻은 문자열을 반환한다. 패턴을 찾지 못하면 string 그대로 반환된다.

* 정규식 문법 (참고 : [정규식 HOWTO](https://docs.python.org/ko/3/howto/regex.html#regex-howto))
  * `[`와 `]` : 일치시키려는 문자 집합인 문자 클래스를 지정하는데 사용
  * `-` : 문자의 범위 나타내기 (`[a-z]`는 소문자 전체)
  * `^` : 여집합 나타내기 (`[^a-z]`는 소문자 제외)
  * `역슬래시(\)` : 모든 메타 문자 이스케이프 처리, 특수 시퀀스 나타내기
    * `\d`는 모든 십진 숫자(=`[0-9]`), `\D`는 모든 비 숫자 문자(=`[^0-9]`)
    * `\w`는 모든 영숫자(=`[a-zA-Z0-9_]`), `\W`는 모든 비 영숫자(=`[^a-zA-Z0-9_]`)
    * `\s`는 모든 공백 문자(=`[\t\n\r\f\v]`), `\S`는 모든 비 공백 문자(=`[^\t\n\r\f\v]`)
  * `*` : 0개 이상과 일치, `+` : 1개 이상과 일치
