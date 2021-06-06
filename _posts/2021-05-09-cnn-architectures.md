---
layout: post
title:  "[통계적 기계학습] 8. CNN architectures"
subtitle:   "ImageNet Challenge 를 통해 발전한 여러 CNN 구조들에 대해 알아보자"
categories: "DeepLearning"
tags: [CNN]
use_math: true
---

# CNN architectures

- 목차
    - [목표](#목표)
    - [AlexNet (2012)](#alexnet-2012)
    - [VGGNet (2014)](#vggnet-2014)

## 목표
* 지금까지 여러가지 블록들을 배웠다. 이를 어떻게 조합해야할지 알아보자.
* ImageNet Classification Challenge 에서 1등한 솔루션들

## AlexNet (2012)
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

## VGGNet (2014)
* VGG Design rules
  * All conv are 3x3 stride 1 pad 1
    * Input 과 Output 의 width, height size 가 같다.
    * Motivation : Two 3x3 conv has same receptive field as a single 5x5 conv, but has fewer parameters and takes less computation!
  * All max pool are 2x2 stride 2
    * 가로세로 픽셀 수는 pooling 때만 바뀐다. (1/2 downsampling)
  * After pool, double #channels
    * Motivation : Conv layers at each spatial resolution take the same amount of computation!
  * Network has 5 convolutional stages
    * Stage 1 : conv-conv-pool
    * Stage 2 : conv-conv-pool
    * Stage 3 : conv-conv-pool
    * Stage 4 : conv-conv-conv-\[conv\]-pool
    * Stage 5 : conv-conv-conv-\[conv\]-pool
    * (VGG-19 has 4 conv in stages 4 and 5)
* 
