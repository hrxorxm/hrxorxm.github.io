---
layout: post
title:  "[통계적 기계학습] 8. CNN architectures"
subtitle:   "ImageNet Challenge 를 통해 발전한 여러 CNN 구조들과 그 이후에 대해 알아보자"
categories: "DeepLearning"
tags: [CNN]
use_math: true
---

# CNN architectures

- 목차
    - [목표](#목표)
    - [AlexNet (2012)](#alexnet-2012)
    - [VGGNet (2014)](#vggnet-2014)
    - [GoogLeNet (2014)](#googlenet-2014)
    - [ResNet (2015)](#resnet-2015)
    - [ResNeXt (2017)](#resnext-2017)
    - [SqueezeNet (2017)](#squeezenet-2017)
    - [DenseNet (2017)](#densenet-2017)
    - [MobileNets (2017)](#mobilenets-2017)
    - [Neural Architecture Search (2017)](#neural-architecture-search-2017)

## 목표

* 지금까지 여러가지 블록들을 배웠다. 이를 어떻게 조합해야할지 알아보자.
* ImageNet Classification Challenge 에서 1등한 솔루션들

## [AlexNet (2012)](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

* 배경
  * 2011년도까지 : 룰베이스, 피쳐추출, linear classfier
  * 2012년 : AlexNet
* Overview
  * 227 x 227 inputs
  * 5 Convolutional layers : 이때까지 중 깊은 편
  * Max pooling
  * 3 fully-connected layers
  * ReLU nonlinearities : ReLU activation 을 사용한 첫번째 neural network
* 추가적인 내용
  * Local response normalization : batch normalization 의 옛날 버전, 지금은 쓰이지 않음
  * 2 GPUs / only 3GB of memory each

## [VGGNet (2014)](https://arxiv.org/abs/1409.1556)

* VGG Design rules
  * `All conv are 3x3 stride 1 pad 1`
    * Input 과 Output 의 width, height size 가 같다.
    * Motivation : Two 3x3 conv has same receptive field as a single 5x5 conv, but has fewer parameters and takes less computation!
  * `All max pool are 2x2 stride 2`
    * 가로세로 픽셀 수는 pooling 때만 바뀐다. (1/2 downsampling)
  * `After pool, double #channels`
    * Motivation : Conv layers at each spatial resolution take the same amount of computation!
  * `Network has 5 convolutional stages`
    * Stage 1 : conv-conv-pool
    * Stage 2 : conv-conv-pool
    * Stage 3 : conv-conv-pool
    * Stage 4 : conv-conv-conv-\[conv\]-pool
    * Stage 5 : conv-conv-conv-\[conv\]-pool
    * (VGG-19 has 4 conv in stages 4 and 5)

## [GoogLeNet (2014)](https://arxiv.org/abs/1409.4842)

* 목표 : `Focus on Efficiency`
  * 파라미터수, 메모리 사용, 계산량 등을 전반적으로 효율적으로 만들기 위한 고민을 많이 한 구조
* `Aggressive Stem`
  * 고해상도의 convolution 연산을 줄이자
    * Start 부분에 high resolution image에 convolution 계산을 해야하니까 상당히 많은 floating point operation 이 집중이 되어있음
    * 따라서 빠르게 downsample 하려고 함
  * 단 3번의 convolution 으로 크기를 **224 -> 28** 로 줄임
* `Inception Module`
  * VGG 가 kernel size 를 3x3 conv 로 고정하는 것과는 달리, 여러 사이즈의 conv 를 통과한 후 결과들을 이어붙인다.
  * `1x1 conv` 사용
    * fully-connected layer를 한번 거친 것과 같은 효과 (약간의 비선형성)
    * 1x1 "Bottleneck" layer 라고 불리기도 한다고 함
    * resnet에서도 나올 예정
* `Global Average Pooling`
  * 끝에 있는 FC layer 때문에 learnable parameter 가 많이 생겼다.
  * 따라서 Global average pooling 을 사용하여 FC layer 로 들어갈 때의 variable 개수를 줄여버린다.
* (추가) `Auxiliary Classifiers`
  * Batch normalization 개념이 나오기 전이라 깊은 네트워크 training 이 어려웠다.
  * 중간중간 결과를 빼서 loss term 에 contribute 하도록 함
  * Batch normalization 이후에는 더이상 필요 없어짐

## [ResNet (2015)](https://arxiv.org/abs/1512.03385)

* 기존 문제점
  * Batch normalization 이후에 10개 이상의 layer를 학습시킬 수 있게 되었다.
    * 대부분의 사람들이 더 깊은 네트워크를 만들면 일부 layer 가 identity function 처럼 되고, 적은 수의 레이어와 비슷해질 것이라고 생각함
    * 층을 많이 쌓은 모델이 층을 적게 쌓은 모델을 포함하는 개념이 된다고 생각함(재현할 수 있다.)
    * 따라서 층이 많은 모델은 적어도 층이 적은 모델과 비슷한 성능이 나와야 한다고 기대함
  * **하지만 여전히 깊은 모델은 얕은 모델보다 성능이 안 좋았음**
    * 오버피팅도 아니고 언더피팅이었음 (학습이 제대로 안 된다는 뜻)
  
* 저자의 가설
  * Hypothesis : `Optimization` 문제다! (identity function 도 쉽게 못 배운다)
  * Solution : **identity function 을 더 쉽게 배울 수 있도록** 네트워크를 조절하자
  
* `Residual Networks`
  * `Plain block`
    * $X$ -> conv -> relu -> conv -> $H(X)$
    
  * `Residual Block`
    * $X$ -> conv -> relu -> conv -> $F(X)$
    * $H(X) = F(X) + X$
    * 직관 1
      * 만약 conv 가 다 0이 되면 결과가 identity function 계산한게 됨
      * 이전 구조보다는 중간중간 노드가 identity function을 잘 배울 수 있게하여 얕은 네트워크를 더 잘 모방할 수 있게 되기를 기대함
    * 직관 2
      * computational graph 관점
      * 더하기 노드는 upstream gradient 를 copy 해서 아래로 보내기 때문에, 중간에 vanishing gradient 문제도 어느정도 경감시킬 수 있다.
    
  * `Residual Networks`
    * A stack of many residual blocks
    * Regular design, like VGG
      * 각 residual block 은 두 개의 **3x3 conv** 를 사용한다.
    * Network is divided into stages
      * 여러 개의 stage 로 이루어짐
      * **각 stage 시작할 때 stride-2 conv 를 이용**하여 1/2 로 downsampling 하고, channel 을 2배로 만든다.
    * Uses the same aggressive stem as GoogLeNet
      * **256 -> 56** 으로 빠르게 줄어든다.
    * Uses global average pooling like GoogLeNet
      * FC layer 에 parameter 수가 많아지는 문제를 경감시킴
    
  * 모델 비교
    | | ResNet-18 | ResNet-34 |
    | :---: | :---: | :---: |
    | Stem | 1 conv layer | 1 conv layer |
    | Stage 1 (C=64) | 2 residual block = 4 conv | 3 residual block = 6 conv |
    | Stage 2 (C=128) | 2 residual block = 4 conv | 4 residual block = 8 conv |
    | Stage 3 (C=256) | 2 residual block = 4 conv | 6 residual block = 12 conv |
    | Stage 4 (C=512) | 2 residual block = 4 conv | 3 residual block = 6 conv |
    | FC layer | Linear | Linear |
    | GFLOP | 1.8 | 3.6 |
    
  * `Bottleneck Block`
    * 3x3 Conv 앞 뒤에 1x1 Conv 를 넣어서 계산량을 줄이도록 함
    * ResNet-50 부터는 Bottleneck block 이용
  
* [Identity Mappings in Deep Residual Networks (2016)](https://arxiv.org/abs/1603.05027)
  * Batch Norm -> ReLU -> Conv -> Batch Norm -> ReLU -> Conv 순서
  * 그렇게 큰 차이는 없다. 실제로 많이 쓰이지는 않는다.

## [ResNeXt (2017)](https://arxiv.org/abs/1611.05431)

* G parallel pathways
  * Bottleneck block을 (inception module 처럼)여러 개 통과시킨 후 sum
  * 같은 계산량으로 만들고 싶다면 $9Gc^2 + 8GCc - 17C^2 = 0$을 만족하는 $c$를 찾아서 Bottleneck block의 3x3 conv의 c로 사용한다.
* Grouped Convolution
  * 아이디어 : $C_{in}$을 기준으로 분리해서 그룹을 나눠서 학습시킨 후 $C_{out}$으로 Concat 하는 방식
  * GPU에서 parallel하게 계산하기 용이함
  * PyTorch - Conv2d(..., groups=1,...) 에서 그룹 수를 지정할 수 있다.
* Bottleneck block에서 3x3 conv를 groups=G로 설정하면 ResNext block이 된다.
* ResNet에서 각자의 Bottleneck block을 grouped convolution으로 바꾼 것을 ResNeXt라고 한다.

## [SqueezeNet (2017)](https://arxiv.org/abs/1709.01507)

* Squeeze-and-Excitation Networks
  * SE-ResNet Module : Residual block 안에 뭔가를 추가해서 넣음
  * 2017년 ILSVRC 1등 : ResNext-152-SE
  * 2017년 이후, 데이터셋이 캐글로 옮겨지면서 ImageNet Competition은 더이상 진행되지 않음

## [DenseNet (2017)](https://arxiv.org/abs/1608.06993)

* Densely Connected Neural Networks
  * Dense Block : Dense blocks where each layer is connected to every other layer in feedforward fashion
* 특징
  * alleviates vanishing gradient
  * strengthens feature propagation
  * encourages feature reuse

## [MobileNets (2017)](https://arxiv.org/abs/1704.04861)

* 지금까지 발전해온 방향 : accuracy 가 최우선
  * downsampling을 빨리
  * 계산, 메모리 효율적인 방향
  * Module 같은 구조를 만든 후에 반복
* Tiny Networks (For Mobile Devices) : top 1 solution 처럼 매우 잘 작동하는 건 아니지만 적당히 잘 작동하면서 매우 적은 수의 연산으로 훈련 가능한 네트워크들
* 추가
  * [ShuffleNet (2018)](https://arxiv.org/abs/1707.01083)
  * [MobileNetV2 (2018)](https://arxiv.org/abs/1801.04381)
  * [ShuffleNetV2 (2018)](https://arxiv.org/abs/1807.11164)

## [Neural Architecture Search (2017)](https://arxiv.org/abs/1611.01578)

* 목표 : 네트워크 디자인 자체도 자동화되게 하고 싶다.
* 방법
  * network를 training시키는 network를 만든다.
  * network 하나 만들 때마다 network를 generate하는 구조가 한번 업데이트 된다.
    * network 하나를 training 시키면 accuracy가 나온다.
    * network를 design하는 network의 gradient descent를 돌린다.
  * 강화학습(reinforce learning)
    * 환경에 action을 날리고 reward를 이용해서 업데이트
    * action : 네트워크 구조 하나
    * reward : training된 네트워크 구조의 accuracy 하나
* 분석
  * 계산량이 엄청나다.
  * 논문에서는 800 GPU로 28일 동안 학습시킴
  * 후속 연구에서는 더 효율적인 Search에 초점을 두고 있다.
  * NAS가 찾아낸 구조들이 사람이 디자인한 모델들보다 훨씬 효율적이다.
