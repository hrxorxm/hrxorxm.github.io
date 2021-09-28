---
layout: post
title:  "[통계적 기계학습] 7. Convolutional Neural Networks"
subtitle:   "합성곱 신경망과 내부 구성 요소에 대해 알아보자 (Self-check 有)"
categories: "Statistical-Machine-Learning"
tags: []
use_math: true
---

# Convolutional Neural Networks

- 목차
    - [목표](#목표)
    - [Convolution Layer](#convolution-layer)
      - [Summary](#summary)
    - [Pooling Layer](#pooling-layer)
    - [Batch Normalization](#batch-normalization)
    - [Self-check](#self-check)

## 목표
* Problem
  * So far our classifiers don't respect the spatial structure of images!
  * 지금 까지는 공간적으로 연관이 있는 이미지의 특성을 무시하고, 한줄 벡터로 펴서 학습시켰다. (fully connected neural network)
  * 이미지의 특성(주변 픽셀이 비슷함)을 최대한 활용하고 싶다.
* Solution
  * Define new computational nodes that operate on images!
  * $W$ 에 특정한 구조를 부여하여 공간적인 의존성을 반영할 수 있도록 하자.
* 지금까지 배운 것 : Fully-Connected Layers, Activation Function
* 오늘 배울 것 : Convolution Layers, Pooling Layers, Normalization

## Convolution Layer
### Comparison
* `Fully-Connected Layer`
  * 계산 : 주어진 이미지 하나의 전체 부분과 내적
  * "MLP" learns Bank of whole image templates
* `Convolution Layer`
  * 계산 : 필터를 **a small chunk of the image** 에 대해서만 내적
  * **convolve** the filter with the image, i.e. "slide over the image spatially, computing **dot products**"
  * "First-layer conv filter" learns local image templates
    * ex) AlexNet : Often learns oriented edges (엣지), opposing colors (보색)

### Summary
* Input
  * $N \times C_{in} \times H \times W$ Batch of images (preserve spatial structure)
* Convolution Layer
  * Hyperparameters
    * Kernel size : $K_H \times K_W$
    * Number filters : $C_{out}$
    * Padding : $P$
    * Stride : $S$
  * Weight matrix : $C_{out} \times C_{in} \times K_H \times K_W$
    * $C_{out}$ filters, each $C_{in} \times K_H \times K_W$
  * Bias vector : $C_{out}$
* Output
  * $N \times C_{out} \times H\prime \times W\prime$ Batch of outputs
    * $C_{out}$ activation maps, each $1 \times H\prime \times W\prime$
    * $H\prime = (H - K + 2P)/S + 1$
    * $W\prime = (W - K + 2P)/S + 1$
  * Parameters per filter : $C_{in} \times K_H \times K_W + 1$
  * Number of learnable parameters : $C_{out} \times (C_{in} \times K_H \times K_W + 1)$
  * Number of multiply-add operations : $(C_{in} \times K_H \times K_W)\times(C_{out} \times H\prime \times W\prime)$
* Common settings

### A closer look at spatial dimensions
* In general
  * Input : $W$, Filter : $K$
  * Output : $W - K + 1$
  * Problem : Feature maps "shrink" with each layer!
  * Solution : **padding** (Add zeros around input)
* `Padding`
  * Input : $W$, Filter : $K$, Padding : $P$
  * Output : $W - K + 1 + 2P$
  * 보통 $P = (K -1) / 2$ 로 하여, input/output size 가 같도록 설정한다.
    * `Receptive Fields`
      * 생물학적 용어 : 시신경 하나가 볼 수 있는 바깥 영역
      * Output의 B영역이 Input의 A영역을 보고있다. A는 B의 receptive field
        * receptive field in the input
        * receptive field in the previous layer
      * $L$ 번째 layer 의 receptive filed size : $1 + L * (K - 1)$
  * Problem : For large images we need many layers for each output to "see" the whole image
  * Solution : **Downsample** inside the network
* `Stride` (Downsample : 뛰는 간격을 넓히기)
  * Input : $W$, Filter : $K$, Padding : $P$, Stride : $S$
  * Output : $(W - K + 2P)/S + 1$

### Stacking Convolutions
* 주의 : convolution layer 를 여러 개 쌓기만 하면 linear transform 을 여러 번 하는 것 밖에 안된다. non-linear 연산을 끼워넣어야 한다.
* Example : Input -> Conv -> **ReLU** -> First hidden layer -> ...

### Other types of convolution
* 1D Convolution
  * Input : $C_{in} \times W$
  * Weights : $C_{out} \times C_{in} \times K$
* 2D Convolution (지금까지 살펴본 기본 ConvNet)
  * Input : $C_{in} \times H \times W$
  * Weights : $C_{out} \times C_{in} \times K \times K$
* 3D Convolution
  * Input : $C_{in} \times H \times W \times D$
  * Weights : $C_{out} \times C_{in} \times K \times K \times K$

## Pooling Layer
* Another way to **downsample**
* Hyperparameters
  * Kernel Size
  * Stride
  * Pooling function
* Max Pooling
  * Introduces **invariance** (불변성) to small spatial shifts
    * 이미지 자체가 공간적으로 조금 움직여도 크게 변하지 않는다는 특성이 있다.
  * No learnable parameters

### Summary
* Input
  * $C_{in} \times H \times W$
* Pooling layer
  * Hyperparameters
    * Kernel size : $K$
    * Stride : $S$
    * Pooling function (max, avg)
* Output
  * $C \times H\prime \times W\prime$
    * $H\prime = (H - K)/S + 1$
    * $W\prime = (W - K)/S + 1$
  * Number of learnable parameters : None!
* Common settings

### Convolutional Networks
* Classic architecture : [Conv, ReLU, Poll] X N, flatten, [FC, ReLU] X N, FC
  * Example : LeNet-5
  * Spatial size **decreases** (using pooling or strided conv) (공간적인 사이즈가 줄어든다.)
  * Number of channels **increases** (total "volume" is preserved!)

## Batch Normalization
* Problem : Deep Networks very hard to train
  * $L(W)$ 가 non-convex in $W$
  * $W$ 에 대한 고차원 (LeNet-5 도 100만 차원 이상)
  * `vanishing gradient` : Layer 가 많아질수록 $L$ 에 대한 앞쪽 layer weight 들의 gradient 가 0 에 너무 가까워짐..

* `Normalization`
  * ${\hat{x}}_{i,j} = \frac{x_{i,j} - \mu_{j}}{\sqrt{\sigma_{j}^{2} + \epsilon}}$
  * Normalization 후에 학습이 좀 더 빨리 된다는 것을 발견
  * 보통 Batch Normalization, Layer Normalization 등..

* `Batch Normalization`
  * imagine (deep) neural network
  * let's focus on one layer and its input
    * input : $X \in R^{N \times D}$
      * N : Batch size
      * D : Dimension
    * layer : linear
  * 문제 의식 : `internal covariance shift`
    * Batch 마다 x 의 분포가 같지 않은 현상
      * Batch마다 Mean vector 랑 covariance 가 달라짐
    * 학습 과정에서 batch 마다 activation 값들의 분포가 다르다.
      * 수학적으로 증명할 수는 없지만, 이럴수록 gradient 계산이 잘 안되고 loss 가 잘 안 떨어지더라(decrease)
      * 이것이 weight 의 학습을 방해하는것 아닌가
  * 방법
    * Input 이 들어오면 바로 layer 에 넣는 것이 아니라 normalization 후 넣는다.
    * Input : $X \in R^{N \times D}$
    * $\mu_{j} = \frac{1}{N} \sum_{i=1}^{N} x_{i,j}$
      * Per-channel mean (변수별 평균), shape is $D$
    * $\sigma_{j}^{2} = \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_{j})^2$
      * Per-channel std (변수별 표준편차), shape is $D$
    * $\hat{x}_{i,j} = \frac{x_{i,j} - \mu_{j}}{\sqrt{\sigma_{j}^{2} + \epsilon}}$
      * Normalized $x$, shape is $N \times D$
      * 분산이 0이 되는 것을 막기 위해서 $\epsilon$ 을 더함
    * Problem : input이 zero-mean, unit variance 인 것이 어려운 제약조건 아닌가?
      * $\hat{X} \cdot \gamma + \beta$ 를 하고, $\gamma, \beta$ 는 learnable 하게 만들면 된다.
      * Learnable scale and shift parameters : $\gamma, \beta \in R^{D}$
      * Learning $\gamma = \sigma, \beta = \mu$ will recover the identity function!
      * 따라서, $y_{i,j} = \gamma_{j} {\hat{x}}_{i,j} + \beta_{j}$
        * Output, shape is $N \times D$

* **Batch Normalization : Test-Time**
  * Problem : Estimates depend on minibatch; can't do this at test-time!
  * test time 때에는 batch 의 개념이 모호하기 때문에 training time 때 저장해둔 mean 과 std 를 사용한다. (저장하는 방식은 다양)
    * $\mu_{j}$ : (Running) average of values seen during <U>training</U>
    * $\sigma_{j}^{2}$ : (Running) average of values seen during <U>training</U>
  * 표준화시키는 작업이 linear operation 이기 때문에, 이전 FC 나 Conv 레이어도 linear operation 이니까 둘을 합쳐서 저장해주면, test time 때 계산 부담이 추가적으로 생기지는 않을 것이다.
* **Batch Normalization for ConvNets**
  * Problem : 그림 구조를 보존하는 batch normalization 은 없을까?
  * Batch Normalization for fully-connected networks
    * $x : N \times D$
    * $\mu, \sigma : 1 \times D$
    * $\gamma, \beta : 1 \times D$
    * $y = \gamma (x - \mu) / \sigma + \beta$
  * Batch Normalization for convolutional networks (Spatial Batchnorm, BatchNorm2D)
    * $x : N \times C \times H \times W$
    * $\mu, \sigma : 1 \times C \times 1 \times 1$
    * $\gamma, \beta : 1 \times C \times 1 \times 1$
    * $y = \gamma (x - \mu) / \sigma + \beta$
* **Batch Normalization 의 위치**
  * Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity.
* Batch Normalization 논문에서 실험으로 밝힌 제한적인 사실 (이론 X)
  * **장점**
    * Inception 과 같은 Deep networks 도 더 훈련이 빠르고 쉽게 만들어준다.
    * learning rate (step size)를 크게 하는 것이 가능해져서 더 빨리 수렴하도록 한다.
    * 초기값 세팅에서 덜 민감해졌다. (robust to initialization)
    * 일종의 훈련을 방해하는 요소가 될 수 있다. (acts as regularization)
    * test time 때 추가적으로 만들어지는 계산노드도 없다.
  * **단점** (한계)
    * 이론적으로 밝히지 못함 (Not well-understood theoretically (yet))
    * batch normalization 의 로직이 training 때와 test 때 달라서, neural network 를 짤 때, 많은 버그의 소스(원인)가 될 수 있다.

* `Layer Normalization`
  * Batch Normalization for fully-connected networks
    * 방법 : each dimension 별로, sample 에 대해서 mean과 std로 표준화시킨 것
    * $x : N \times D$
    * $\mu, \sigma : 1 \times D$
    * $\gamma, \beta : 1 \times D$
    * $y = \gamma (x - \mu) / \sigma + \beta$
  * **Layer Normalization** for fully-connected networks
    * 방법 : 각 sample 에 대해서, dimenstion 별 mean과 std로 표준화시킨 것
      * Same behavior at train and test!
      * Used in RNNs, Transformers
    * $x : N \times D$
    * $\mu, \sigma : N \times 1$
    * $\gamma, \beta : N \times 1$
    * $y = \gamma (x - \mu) / \sigma + \beta$  
* `Instance Normalization`
  * Batch Normalization for convolutional networks
    * $x : N \times C \times H \times W$
    * $\mu, \sigma : 1 \times C \times 1 \times 1$
    * $\gamma, \beta : 1 \times C \times 1 \times 1$
    * $y = \gamma (x - \mu) / \sigma + \beta$
  * **Instance Normalization** for convolutional networks
    * $x : N \times C \times H \times W$
    * $\mu, \sigma : N \times C \times 1 \times 1$
    * $\gamma, \beta : N \times C \times 1 \times 1$
    * $y = \gamma (x - \mu) / \sigma + \beta$
* `Group Normalization`
  * 각 sample 에 대해서, color channel 몇 개씩 모아서 mean, std 를 구하는 것


***

# Self-check

* 다음 문제들을 단답형 (풀이 필요 없음) 또는 서술형 (수기풀이 필요)으로 답하세요. 2번과 3번만 서술형입니다. 
* 문제 출처는 Coursera의 Convolutional Neural Networks (in Deep Learning Specialization led by Andrew Ng)의 1주차 퀴즈입니다.  
* (혼동 주의) 아래 문제들은 우리 수업과 달리 image tensor를 (가로픽셀수) * (세로픽셀수) * (채널 수)로 표현하고 있습니다. 수업에서는 (채널 수) * (세로픽셀수) * (가로픽셀수)를 사용중입니다. 

## 문제1 (단답형)
![image](https://user-images.githubusercontent.com/35680202/114643245-4229b800-9d10-11eb-9258-755be3985ec5.png)
* 답 : Detect vertical edges

## 문제2 (서술형)
![image](https://user-images.githubusercontent.com/35680202/114651498-0e09c380-9d1f-11eb-8bab-e31f02c0cb59.png)
* 답 : (3 X 300 X 300 + 1)*100 = 27,000,100

## 문제3 (서술형)
![image](https://user-images.githubusercontent.com/35680202/114651542-21b52a00-9d1f-11eb-94c9-6e5344023432.png)
* 답 : 100*(3 X 5 X 5 + 1) = 7600

## 문제4 (단답형)
![image](https://user-images.githubusercontent.com/35680202/114651570-3396cd00-9d1f-11eb-9bb2-679d00250fb6.png)
* 답 : 29 X 29 X 32

## 문제5 (단답형)
![image](https://user-images.githubusercontent.com/35680202/114651603-44dfd980-9d1f-11eb-8ae5-25ef90b67709.png)
* 답 : 19 X 19 X 8

## 문제6 (단답형)
![image](https://user-images.githubusercontent.com/35680202/114651634-50cb9b80-9d1f-11eb-8709-8c712f86c10c.png)
* 답 : P = 3

## 문제7 (단답형)
![image](https://user-images.githubusercontent.com/35680202/114651680-617c1180-9d1f-11eb-9c67-32d3898b690d.png)
* 답 : 16 X 16 X 16

## 문제8 (단답형)
![image](https://user-images.githubusercontent.com/35680202/114651708-6e990080-9d1f-11eb-83e5-fffd5e4a2b75.png)
* 답 : False