---
layout: post
title:  "[공돌이의 수학정리노트] Gaussian Mixture Model"
subtitle:   "Gaussian Mixture Model에 대한 강의 내용 정리"
categories: "Speech-Recognition"
tags: []
use_math: true
---

# GMM과 EM 알고리즘

> 강의 출처 : [가우시안 혼합 모델 (GMM) & E-M 알고리즘](https://youtu.be/NNwkDi-2xVQ)

> 강의 노트 : [공돌이의 수학정리노트 - GMM과 EM 알고리즘](https://angeloyeo.github.io/2021/02/08/GMM_and_EM.html)

* GMM(Gaussian Mixture Model) : 가우시안 혼합 모델
* EM(Expectation Maximization) 알고리즘

## 사전지식

* k-means 알고리즘
* 최대우도법
* 베이즈정리의 의미
* 나이브 베이즈 분류기 (혹은 Maximum A Posteriori)

## 최대우도법

* 최대우도법(Maximum Likelihood Estimation)
  * 어떤 데이터를 관찰하고 데이터에 맞는 분포를 상정하여 그 분포의 모수를 추정하는 방법
  * 우도를 정의한 후 이 우도를 최대로 만드는 분포를 찾는 방식
    * 우도 : 각 데이터 샘플에서 후보 분포에 대한 높이를 다 곱한 것
    * $\Pi_{i=1}^{m} p(x^{(i)} \| \theta)$
  * 예시 : 정규분포
    * $\hat{\mu} = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$
    * $\hat{\sigma}^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \hat{\mu})^2$
    * 이 때 가장 likelihood 가 높다.

## 가우시안 혼합 모델 - 간단ver

* 가우시안 혼합 모델(GMM)
  * 문제 : 두 라벨로 구별할 수 있는 데이터는 MLE로 두 라벨의 확률 분포 추정 가능, 그러나 라벨이 주어지지 않는다면?
    * 라벨이 있으려면 분포가 있어야 하고, 분포를 얻으려면 라벨이 필요 (닭과 달걀 문제)
  * 해결 : 분포를 랜덤하게 설정하고 시작하자. 즉, 두 개의 분포가 있을 것이라 상정하고, 각 분포에 대한 모수를 랜덤하게 선정
    * 이제 이 분포를 이용해서 각 데이터의 라벨을 구하고, 라벨을 이용해서 다시 각 그룹의 모수를 다시 추정해준다. (반복)

## 가우시안 혼합 모델 - 수식ver

* EM 알고리즘
  * Expectation과 Maximization 과정을 반복적으로 수행하는 알고리즘
  * Expectation : 각 데이터에 라벨을 부여하는 과정
  * Maximization : 각 그룹의 모수를 재 계산하는 과정
  * 주의 : local maxima에 빠질 수 있으니 이럴 때는 initialization을 다시 해준 후 돌리기
* E-step
  * $w_j^{(i)} := p(z^{(i)}=j \| x^{(i)}; \phi, \mu, \Sigma)$ : i 번째 데이터가 j 그룹에 들어갈 확률
    * $\phi, \mu, \Sigma$는 이미 주어진 파라미터
    * $\phi$ : 각 라벨별 평균 소속 확률
    * $\mu, \Sigma$ : 평균과 공분산
  * $p(z^{(i)}=j \| x^{(i)}; \phi, \mu, \Sigma)$
    * $= \frac{p(x^{(i)} \| z^{(i)}=j; \mu, \Sigma) p(z^{(i)} = j; \phi)}{p(x^{(i)}; \phi, \mu, \Sigma)}$ (베이즈 정리에 의해)
      * $p(x^{(i)}; \phi, \mu, \Sigma)$ : evidence
      * $p(z^{(i)} = j; \phi)$ : prior
      * $p(x^{(i)} \| z^{(i)}=j; \mu, \Sigma)$ : likelihood
    * $= \frac{p(x^{(i)} \| z^{(i)}=j; \mu, \Sigma) p(z^{(i)} = j; \phi)}{\sum_{k=1}^{l} p(x^{(i)} \| z^{(i)} = l; \mu, \Sigma) p(z^{(i)} = j; \phi)}$
* M-step
  * 최대우도법을 이용하여 모수를 계산해주기만 하면 된다.
  * 즉, 평균, 표준편차만 계산해주면 된다.
