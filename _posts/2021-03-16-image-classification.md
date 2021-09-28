---
layout: post
title: "[통계적 기계학습] 2-1. Image classification"
subtitle: "이미지 분류 문제에 대한 정의와 잘 알려진 데이터셋에 대해 알아보자"
categories: "Statistical-Machine-Learning"
tags: []
use_math: true
---

# 이미지 분류의 개요

- 목차
  - [이미지 분류란?](#이미지-분류의-개요)
    - [1. 용어](#1-용어)
    - [2. 목표](#2-목표)
    - [3. 방법](#3-방법)

## 1. 용어
* `Image Classification` : 이미지가 주어졌을 때, 무엇을 뜻하는지 (ex. 5개 중 하나로) 분류하는 것

* `Semantic Gap` : 컴퓨터는 이미지를 픽셀로 추상적으로 인식하고, 인간은 의미로 인식한다.
  * Viewpoint Variation : 조금만 각도가 달라져도 픽셀값이 달라지기 때문에 컴퓨터는 다른 것이라고 인식한다.
  * Intraclass Variation : 같은 클래스 내에서도 다르게 생길 수도 있다.
  * Fine-Grained Categories : 클래스 분류 후 세부 종류도 알아내는 것
  * Background Clutter : 배경과 구분이 어려운 것
  * Illumination Changes : 빛이 어디서 비추는지에 영향을 받지는 않는지
  * Deformation : 다양한 모습, 자세
  * Occlusion : 이미지 자체에서 근거가 약한 상태 (일부 모습만 노출되는 등)

## 2. 목표
* 목표 : Robust한 Image Classifier를 만들고 싶다!
  * Robustness : 외부 환경에 굴하지 않고 안정적으로 기기가 동작하는 것

* Image Classification 을 공부해야 하는 이유
  * 유용하다. 상업적인 가치가 크다.
  * 이미지 분류기를 다른 작업을 위해 사용할 수 있다. (**building block for other vision tasks**)
    * Object Detection : 이미지가 주어졌을 때, 객체가 있는 네모박스를 아웃풋을 주는 것
    * Image Captioning : 그림과 단어가 같이 주어지고, 이미지를 설명하는 문장을 만들어내는 것
    * Playing Go : 알파고

## 3. 방법
* Image Classifier를 만드는 방법
  * 하드코드 알고리즘 : 어떻게 룰로 만들지가 불분명하다.
    * ex) Edge를 찾아서 몇 개 있는지 세서 분류...?
      * 한계가 있다. robust하지 않다.
      * 사람의 개입을 최소로 하고싶다. (사람의 선험적인 지식이 필요하지 않도록)
  * `Machine Learning : Data-Driven Approach`
    * 이미지와 레이블의 데이터셋을 모아서 머신러닝 기법으로 학습시킨 후 새로운 이미지로 평가하기
    * 가장 중요한 **데이터 수집**을 통해 직접 실행해본 사람들 덕분에 발전할 수 있었다.

## 4. 잘 알려진 데이터셋
### < 적당한 규모 >
* `MNIST`
  * $\{(x_i,y_i)\}^{50k}_{i=1}$, $x_i \in R^{28 \times 28}$, $y_i \in \{0, 1, \dotsc, 9\}$
  * **새로운 방법론을 테스트**할 때 많이 사용한다. (첫 실험)
  * Deep Learning 이 아닌 Shallow Machine Learning 기법인 Random Forest 조차  좋은 성능을 낸다. 
* `CIFAR10`
  * $\{(x_i,y_i)\}^{50k}_{i=1}$, $x_i \in R^{32 \times 32 \times 3}$, $y_i \in \{0, 1, \dotsc, 9\}$
  * 로컬 컴퓨터로 돌릴 수 있을 정도의 적당한 크기
* `CIFAR100`
  * $\{(x_i,y_i)\}^{50k}_{i=1}$, $x_i \in R^{32 \times 32 \times 3}$, $y_i \in \{0, 1, \dotsc, 99\}$
  * 20 superclasses with 5 classes each : class 간에 계층구조가 있음
### < 큰 규모 >
* `ImageNet` (자주 사용하는 버전)
  * $\{(x_i,y_i)\}^{50k}_{i=1}$, $x_i \in R^{256 \times 256 \times 3}$, $y_i \in \{0, 1, \dotsc, 999\}$
  * GPU 여러 개의 환경에서 하루 이틀은 학습시켜야하는 크기
  * 방법론들을 비교할 때 벤치마크 데이터셋으로 많이 쓰임
  * Performance metric : 상위 5개 중에 정답이 들어있으면 맞춘걸로
* `MIT Places`
  * $\{(x_i,y_i)\}^{8M}_{i=1}$, $x_i \in R^{256 \times 256 \times 3}$, $y_i \in \{0, 1, \dotsc, 365\}$
* `Omniglot`
  * $\{(x_i,y_i)\}^{32k}_{i=1}$, $x_i \in R^{256 \times 256 \times 3}$, $y_i \in \{0, 1, \dotsc, 1622\}$
  * 샘플 사이즈가 적은 데이터셋으로 테스트하기 위해 사용
### < 이 외 >
* 캐글
* AI 바우처, AI 공공데이터
