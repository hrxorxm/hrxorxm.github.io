---
layout: post
title:  "[Boostcamp AI Tech] 4주차 - Image Classification"
subtitle:   "Naver Boostcamp AI Tech Level1 P Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level1-P-Stage]
use_math: true
---

# 부스트캠프 4주차

## [1강] Competition with AI Stages!

* Competition : 주어진 데이터를 이용해 원하는 결과를 만들기 위한 가장 좋은 방법 찾기
* Competition Details
  * 💡**Problem Definition**(문제 정의)💡
    * 지금 풀어야 할 문제가 무엇인가
    * Input과 Output은 무엇인가
    * 이 솔루션은 어디서 어떻게 사용되는가
  * Overview : 방향성 (중요!)
    * ex) Jigsaw Unintended Bias in Toxicity Classification : 그냥 NLP 악성 댓글 분류가 아니라 동음이의어에 대한 문제(실제 풀고자 하는 문제)도 고려해야 함
  * Data Description
    * 데이터 스펙 요약 : 파일 형태, 메타데이터 소개 및 설명
  * Notebook
    * 데이터 분석, 모델 학습, 추론 과정을 서버에서 가능
  * Submission & Leaderboard
    * 테스트 예측 결과물 제출 & 순위 확인
  * Discussion
    * 등수를 올리는 것보다, 문제를 해결하고 싶은 마음
    * 문제를 해결하는 과정에서 토의를 통해서 스스로도 성장하는 것이 목표

## [2강] Image Classification & EDA

* EDA(Exploratory Data Analysis) : 탐색적 데이터 분석
  * 주어진 데이터에 대한 본인의 생각을 자유롭게 서술하시오
  * 나의 궁금증, 의문점을 해결하기 위해 데이터를 확인하는 과정
  * 명제가 정해진 후에 검색해보고 도구 선택하기 (손, Python, Excel, ...)
* Image Classification
  * Image : 시각적 인식을 표현한 인공물(Artifact)
  * Image Classification : Image(Input) + Classification Model = Class(Output)
* Baseline
  * 베이스라인 코드가 주어지기 전에 직접 코드를 작성해보고 예시 코드와 비교해보기

## [3강] Dataset

* Data Science : Pre-processing 80%, Model/etc 20%
* Pre-processing
  * Bounding box : 핵심 객체가 있는 곳을 crop하는 것이 좋다.
  * Resize : 작업의 효율화를 위해, 너무 큰 이미지는 줄인다.
  * 도메인, 데이터 형식에 따라 정말 다양한 Case가 존재한다.
    * ex) APTOS Blindness Detection
* Generalization
  * Bias & Variance
    * High Bias : Underfitting
    * High Variance : Overfitting
  * Train / Validation
    * 훈련 셋 중 일정 부분을 따로 분리하여 검증 셋으로 활용
  * Data Augmentation
    * 주어진 데이터가 가질 수 있는 Case(경우)와 State(상태)를 다양하게 만든다.
    * `torchvision.transforms` / `Albumentations`
    * RandomCrop, Flip 등 다양한게 있지만, 우리 도메인에서 의미가 있을지를 고민해보고 적용해보기
    * 항상 좋은 방법은 없으니 실험으로 증명하자

## [4강] Data Generation

* Data Feeding
  * Feed : 대상의 상태를 고려해서 적정한 양을 준다.
  * Data Generator의 성능과 Model의 성능을 고려하여 효율적으로 tuning하기
  * `transform.Compose` 내의 계산 순서에 따라서도 속도차이가 크다 (특히, `Resize`)
* Dataset
  * Vanilla Data를 Dataset으로 변환
* DataLoader
  * 내가 만든 Dataset을 효율적으로 사용할 수 있도록 관련 기능 추가
  * 간단한 실험 : `num_workers` 등
* Dataset과 DataLoader는 분리되는 것이 좋다.

## [5강] Model 1 - Model with Pytorch

* Pytorch : Low-level, Pythonic, Flexibility
* Pytorch 모델의 모든 레이어는 `nn.Module` 클래스를 따른다.
  * `model.modules()`
    * `__init__()` 에서 정의한 또 다른 `nn.Module`
    * 모델을 정의하는 순간 그 모델에 연결된 모든 module을 확인할 수 있다.
  * `forward()`
    * 이 모델이 호출되었을 때 실행되는 함수
    * 모든 `nn.Module`은 `forward()` 함수를 가지므로, 내가 정의한 모델의 `forward()` 실행으로 각각의 모듈의 `forward()`가 실행된다.
  * Parameters
    * 모델에 정의되어 있는 modules가 가지고 있는, 계산에 쓰일 Parameter
    * data, grad, requires_grad 변수 등을 가지고 있다.
* Pythonic : (ex) `model.state_dict()`는 Python 의 Dictionary이다.

## [6강] Model 2 - Pretrained Model

* ImageNet : 대용량의 데이터셋 덕분에 컴퓨터 비전이 발전할 수 있었다.
* Pretrained Model
  * 목적 : 좋은 품질, 대용량의 데이터로 미리 학습한 모델을 내 목적에 맞게 다듬어서 사용
  * `torchvision.models` / `timm`
* Transfer Learning
  * CNN base : Input + CNN Backbone + Classifier(fc) -> Output
  * Backbone 모델의 task와 현재 task의 유사성을 고려하여 학습한다.
    * High Similarity : CNN Backbone을 Freeze하고 Classifer만 학습시킨다. (Feature extraction 방식)
    * Low Similarity : 전체 네트워크를 학습시킨다. (Fine-tuning 방식)
* 추가 자료
  * [7 Tips To Maximize PyTorch Performance](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)
    * DataLoader
      * `num_workers = 4 * num_GPU`
      * `pin_memory = True` & `torch.cuda.empty_cache()`
    * `.cpu()` / `.item()` / `.numpy()` 는 성능이 매우 느리므로 GPU 상에서 계산 그래프에서만 분리하고 싶을 때는 `.detach()` 를 사용하자
    * `.cuda()` 하지 말고, tensor를 만들 때 device를 지정해주자
  * [Most Common Neural Net PyTorch Mistakes](https://medium.com/missinglink-deep-learning-platform/most-common-neural-net-pytorch-mistakes-456560ada037)
    * train / eval 모드 바꿔주는 것 주의하자
    * `backward()` 전에 `zero_grad()` 해주는 것 잊지 말자

## [7강] Training & Inference 1 - Loss, Optimizer, Metric

* Loss : Loss 함수 = Cost 함수 = Error 함수
  * `nn.Module` 을 상속하고 있다. (forward 함수 포함)
  * `loss.backward()` : `required_grad=True`인 모델의 파라미터의 grad 값이 업데이트 된다.
  * Focal loss : Class Imbalance 문제가 있는 경우, 맞출 확률이 높은 Class는 조금의 loss를, 맞춘 확률이 낮은 Class는 Loss를 훨씬 높게 부여
  * Label Smoothing Loss : Class target label을 Onehot 표현으로 사용하기 보다는 Soft하게 표현에서 일반화 성능을 높이기 위함
* Optimizer : 어느 방향으로, 얼마나 움직일지 결정
  * $w' = w - \eta \frac{\partial E}{\partial w}$
    * $\eta$ : 학습률(Learning rate)
    * $\frac{\partial E}{\partial w}$ : 방향
  * LR scheduler : 학습 시에 Learning rate를 동적으로 조절
    * StepLR : 특정 Step마다 LR 감소
    * CosineAnnealingLR : Cosine 함수 형태처럼 LR을 급격히 변경
    * ReduceLROnPlateau : 더 이상 성능 향상이 없을 때 LR 감소
* Metric : 학습된 모델을 객관적으로 평가할 수 있는 지표
  * Classification : Accuracy, F1-score, precision, recall, ROC&AUC
    * Accuracy : Class 별로 밸런스가 적절히 분포하는 경우
    * F1-Score : Class 별 밸런스가 좋지 않아서 각 클래스 별로 성능을 잘 낼 수 있는지 확인 필요
      * [macro vs micro vs weighted vs samples](https://stackoverflow.com/questions/55740220/macro-vs-micro-vs-weighted-vs-samples-f1-score)
  * Regression : MAE, MSE
  * Ranking : MRR, NDCG, MAP

## [8강] Training & Inference 2 - Process

* Training Process
  * `model.train()` : 모델을 train 모드로 만들기 (`Dropout`이나 `BatchNorm` 등은 학습/추론 시 동작이 다르기 때문)
  * `optimizer.zero_grad()` : 각각의 파라미터의 grad를 0으로 만드는 작업
  * `loss = criterion(output, labels)` : criterion이 nn.Module을 상속하고 있기 때문에 forward 함수를 가지고 있고, input 부터 시작해서 계산 그래프가 완성된다.
  * `loss.backward()` : 계산 그래프가 완성된 후 backward를 진행할 수 있다.
  * `optimizer.step()` : 파라미터 업데이트
  * Gradient Accumulation : 여러 개의 배치에서의 loss를 하나의 배치에서 step하듯이 학습하고 싶을 때 응용할 수 있다.
* Inference Process
  * `model.eval()` : 모델을 eval 모드로 만들기
  * `with torch.no_grad()` : inference 시의 계산은 gradient나 파라미터의 변화 없이 진행되어야 한다.
  * 검증(Validation) 확인 후 좋은 결과를 Checkpoint로 저장
  * 최종 Output, Submission 형태로 변환
* Appendix : Pytorch Lightning
  * Keras와 비슷하다. 생산성 측면에서 좋다.
  * 그래도 공부는 Pytorch로 하는 것이 좋다.

## [9강] Ensemble

* Ensemble(앙상블) : 서로 다른 여러 학습 모델을 사용하여 하나의 모델보다 더 나은 성능을 만들고자 하는 것
  * Model Averaging(Voting) : 다른 모델이 같은 에러를 만들어내지 않을 것이기 때문에 잘 동작한다.
    * Hard Voting : 각각 예측 후 다수결
    * Soft Voting : 점수를 모아서 예측
  * Cross Validation : 훈련 셋과 검증 셋을 분리하되, 검증 셋을 학습에 활용하고 싶다.
    * 모델을 평가하는 지표로서 사용될 수 있다.
    * Stratified K-fold Cross Validation : split 시에 class 분포까지 고려
  * TTA(Test Time Augmentation) : 테스트 이미지를 Augmentation 후 모델 추론, 출력된 여러가지 결과를 앙상블
  * 앙상블 효과는 확실히 있지만, 성능과 효율의 Trade-off 가 있다.
* Hyperparameter Optimization
  * Hyperparameter : 시스템 매커니즘에 영향을 주는 주요한 파라미터
    * Hidden Layer 갯수, k-fold
    * Learning rate, Batch size
    * Loss 파라미터, Optimizer 파라미터
    * Dropout, Regularization
  * 시간과 장비가 충분하다면 시도해볼 수 있긴 하다.
  * Optuna : 파라미터 범위를 주고 그 범위 안에서 trials 만큼 시행한다.

## [10강] Experiment Toolkits & Tips

* Training Visualization
  * Tensorboard : 학습 과정을 기록하고, 트래킹할 수 있다.
    * `tensorboard --logdir PATH --host ADDR --port PORT`
    * ADDR : 원격 서버에서 사용시 `0.0.0.0`로 지정 (default: localhost)
    * PORT : AI Stages 서버에서 열어준 포트번호는 `6006`
  * Weight and Bias (wandb) : 딥러닝 로그의 깃허브 같은 느낌
    * wandb login 처음 1회만 하면 사용 가능
    * 파이썬 코드에서 wandb init, log 설정하여 사용하기
* Machine Learning Project
  * Jupyter Notebook
    * 장점
      * 코드를 아주 빠르게 cell 단위로 실행해볼 수 있다.
      * EDA 등 데이터 분석을 할 때 데이터를 로드해놓고 여러가지 작업할 수 있다.
    * 단점
      * 학습 도중 노트북 창이 꺼지면 tracking이 잘 안됨
  * Python IDLE
    * 장점
      * 구현은 한번만, 사용은 언제든, 간편한 코드 재사용
      * 디버깅 툴이 강력하다.
      * 자유로운 실험 핸들링
* Some Tips
  * 다른 사람의 분석을 볼 때는 코드 보다는 설명글(필자의 생각 흐름)을 유심히 보자
  * 코드를 볼 때는 디테일한 부분까지 보자
  * [Paper with Codes](https://paperswithcode.com/) : 최신 논문과 그 코드까지 확인할 수 있다.
  * 공유하는 것을 주저하지 말자. 새로운 배움의 기회가 될 수 있다.
