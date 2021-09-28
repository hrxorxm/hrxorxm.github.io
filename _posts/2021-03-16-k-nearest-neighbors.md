---
layout: post
title:  "[통계적 기계학습] 2-2. k-Nearest Neighbors"
subtitle:   "k-최근접 이웃 알고리즘과 이미지 분류 학습 방법에 대해 알아보자"
categories: "Statistical-Machine-Learning"
tags: []
use_math: true
---

# k-Nearest Neighbors (k-최근접 이웃)

- 목차
  - [k-Nearest Neighbors (k-최근접 이웃)](#k-nearest-neighbors)
    - [일반적인 순서](#일반적인-순서)
    - [k-NN 에서 $f$ 를 만드는 방법 (training 방법)](#k-nn-에서-f-를-만드는-방법-training-방법)
    - [Hyperparameters](#hyperparameters)
    - [결론](#결론)

## 일반적인 순서
* 데이터셋 구성
  * training dataset
    * $\{(x_i,y_i)\}^{N}_{i=1}$, $x_i \in R^{D}$, $y_i \in \{1, \dotsc, C\}$
  * validation dataset
    * $\{(x_i,y_i)\}^{M}_{i=N+1}$
  * test Dataset
    * $\{(x_i,y_i)\}^{L}_{i=M+1}$
* 방법
  1. **f 를 만들어 준다.**
  2. f 에 new $x$ 를 집어 넣는다. $\hat{y} := f(x)$
  3. $\hat{y}$와 $y$ 비교($i=M+1, \dotsc, L$)

## k-NN 에서 $f$ 를 만드는 방법 (training 방법)
1. training set $\{(x_i,y_i)\}^{N}_{i=1}$ 을 전부 기억한다.
2. new $x$가 들어오면,
   1. $x$ 와 $x_1, x_2, \dotsc, x_N$ 까지의 거리들을 전부 계산 : calculate $d(x, x_1), d(x, x_2), \dotsc, d(x, x_N)$
   2. 가장 가까운 ($d(x,x_i) (i=1 \dotsc N)$가 가장 작은) k 개의 index 들에 대하여. $y_i$ 의 majority vote 를 예측값으로 리턴함 ex) $f(x) = 3$ 
* 단, "거리" $d(x, z)$는 "잘 정의된 거리"
  * ex) $x, z \in R^D$ 이면, 다음 거리들 고려 가능
    * **L2 거리** : $d(x, z) = {\| x-z \|}_2 = \sqrt{\sum_{j=1}^{D} (x_j - z_j)^2}$
    * **L1 거리** : $d(x, z) = {\| x-z \|}_1 = \sum_{j=1}^{D} \mid x_j - z_j \mid$
  * 일반적으로 $d(x, z)$ 는 다음 조건들을 만족해야 한다.  (**"metric"**)
    * 모든 $x, z$에 대하여 $d(x, z) = d(z, x)$
    * 모든 $x$ 에 대하여 $d(x, x) = 0$
    * 모든 $x, z$에 대하여 $d(x, z) \geq 0$ (if $x \neq z$, $d(x, z) > 0$)
    * (삼각 부등식) 임의의 $x, z, w$ 에 대하여 $d(x, w) \leq d(x, z) + d(z, w)$

* time complexity
  * training : $O(1)$
  * testing : $O(N*(L-M))$
  * slow training 

* Nearest Neighbor Decision Boundaries (결정경계) : $f$ 가 정해지면 결정됨

## Hyperparameters
### Hyperparameters 란?
* `Distance Metric`, `Hyperparameter k` 를 적절히 선택해야하는 문제
* 학습 데이터로부터 배울 수 있는 것이 아니다.
* 최적의 $k$ 와 $d$ 는 없다. 주어진 데이터셋에 의존해서 언제나 바뀐다.
* hyperparameter 에 따라서 $\hat{y}$ 를 건설하는 rule 이 달라진다.

### Setting Hyperparameters
* 학습 데이터로 그대로 평가해서 hyperparameter를 고른다면?
  * $k=1$ 은 학습 데이터에 대해서 항상 완벽한 결과(자신의 라벨)를 낸다.
* train 과 test 로 나눈다면?
  * 나중에 서비스할 때 어떻게 잘 작동할지 알 수 가 없다. (Wild 환경을 재현해줄 데이터가 남지 않는다.)
* train 과 validation 과 test 로 나눈다면?
  * train 으로 $f_1, f_2, \dotsb$ 등을 만든다.
  * validation 으로 $f_1, f_2, \dotsb$ 등의 성능을 잰다.
  * test 로 실제 프로덕션 때 어느 정도의 성능을 낼지 예측
* Cross-Validation
  * test 는 그대로 두고, train 부분을 fold 로 나눈 다음에 각각 validation 부분을 바꿔서 학습 & 측정 한다.
  * 각 $f_1, f_2, \dotsb$ 의 평균 성능이 가장 좋은 것을 고른다.

### Validation set 의 역할
* 데이터셋 구성
  * training dataset : $\{(x_i,y_i)\}^{N}_{i=1}$
  * `validation dataset` : $\{(x_i,y_i)\}^{M}_{i=N+1}$
    * 여러 방법론의 hyperparameter 비교
  * test dateset : $\{(x_i,y_i)\}^{L}_{i=M+1}$

* 하나의 $f$ 성능을 평가하는 방법 (분류 문제)
  * validation set : $\{(x_i,y_i)\}^{M}_{i=1}$ (training set 이랑 독립적인 데이터셋)
  * true label : $y_i$, predicted label : $\hat{y_i} (= f(x_i))$
  * classification accuracy (정분류율) $= \frac{맞은 i의 수}{M} = \frac{\sum_{i=1}^{M}I(y_i=\hat{y_i})}{M}$
  * miss classification rate (오분류율) $= \frac{틀린 i의 수}{M} = \frac{\sum_{i=1}^{M}I(y_i \neq \hat{y_i})}{M}$
    * 또는 classification error

## 결론
### k-NN은 좋은 분류기인가?
  * 이론적으로는 그렇다.
  * Universal Approximation : training size가 무한대(dimension을 채우는 빽빽한 점들)면, 어떤 true function 이던지 가까워질 수 있다.

### 이미지 데이터에 적용할 때의 이론적인 한계점
  * Curse of Dimensionality (차원의 저주) : dimension(D) 이 커지면 dimension을 비슷한 거리의 점들로 꽉채우기 위한 점들이 기하급수적으로 많이 필요해진다.
  * Very slow at test time
  * Distance metrics on pixels are not informative
    * 픽셀 데이터에 대한 거리 계산이 실제로 무엇을 의미하는가에 대한 의문점이 있을 수 있다.
    * ConvNet features 을 뽑아서 거리를 계산하면 의미가 있을 수 있다. **Nearest Neighbor 자체가 이상한 개념은 아님!**

