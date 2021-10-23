---
layout: post
title:  "[Boostcamp AI Tech] 3주차 - PyTorch"
subtitle:   "Naver Boostcamp AI Tech Level1 U Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level1-U-Stage]
use_math: true
---

# 부스트캠프 3주차

## [1강] Introduction to PyTorch

* 프레임워크를 공부하는 것이 곧 딥러닝을 공부하는 것이다.
* 종류
  * PyTorch(facebook)
    * Define by Run (Dynamic Computation Graph) : 실행을 하면서 그래프를 생성하는 방식
    * 개발과정에서 디버깅이 쉽다. (pythonic code)
  * TensorFlow(Google)
    * Define and Run : 그래프를 먼저 정의한 후 실행시점에 데이터를 흘려보냄(feed)
    * Production, Scalability, Cloud, Multi-GPU 에서 장점을 가진다.
    * Keras 는 Wrapper 다. (high-level API)
* 요약
  * PyTorch = Numpy + AutoGrad + Function

## [2강] PyTorch Basics

* Tensor : 다차원 Arrays를 표현하는 클래스 (numpy의 ndarray와 동일)
  * list to tensor, ndarray to tensor 모두 가능
  * tensor는 GPU에 올려서 사용가능
* Operations : numpy 사용법이 거의 그대로 적용된다.
  * reshape() 대신 view() 함수 사용 권장
  * squeeze(), unsqueeze() 차이와 사용법 익히기
  * 행렬곱은 mm(),matmul() 사용 (matmul은 broadcasting 지원)
  * 내적은 dot() 사용
  * `nn.functional` 에서 다양한 수식 변환 지원
* AutoGrad : 자동 미분 지원
  * tensor(requires_grad=True)로 선언한 후 backward() 함수 사용
  * [A GENTLE INTRODUCTION TO `TORCH.AUTOGRAD`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
  * [PYTORCH: TENSORS AND AUTOGRAD](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html)

## [3강] PyTorch 프로젝트 구조 이해하기

* 목표
  * 초기 단계 : 학습과정 확인, 디버깅
  * 배포 및 공유 단계 : 쉬운 재현, 개발 용이성, 유지보수 향상 등
* 방법
  * OOP + 모듈 => 프로젝트
  * 실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등을 분리하여 프로젝트 템플릿화
* 템플릿 추천
  * [Pytorch Template](https://github.com/victoresque/pytorch-template) <- 실습 진행📌
  * [Pytorch Template 2](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template)
  * [Pytorch Lightning Template](https://github.com/PyTorchLightning/deep-learning-project-template)
  * [Pytorch Lightning](https://www.pytorchlightning.ai/)
  * [Pytoch Lightning + NNI Boilerplate](https://github.com/davinnovation/pytorch-boilerplate)
* 구글 코랩과 vscode 연결하기
  * colab
    * [ngrok](https://ngrok.com/) 가입하기
    * `colab-ssh` 설치하기
    * 토큰 넣어서 `launch_ssh` 실행해서 연결정보 확인하기
  * vscode
    * extension : Remote - SSH install 하기
    * Remote-SSH: Add New SSH Host 에서 `ssh root@[HostName] -p [Port]` 입력
    * Remote-SSH: Connect to Host 에서 위에서 등록한 host 연결하기
    * `cd /content`로 가면 코랩에서 다운받았던 파일들을 볼 수 있다.
    * `/content/pytorch-template/MNIST-example`에서 `python3 train.py -c config.json` 으로 예제를 실행해볼 수 있다.

## [4강] AutoGrad & Optimizer

* 딥러닝 아키텍쳐 : 블록 반복의 연속
  * Layer = Block
* `nn.Module` : Layer의 base class
  * Input, Output, Forward, Backward, parameter 정의
* `nn.Parameter` : Tensor 객체의 상속 객체
  * `nn.Module` 내에서 parameter로 정의될 때, `required_grad=True` 지정하기
  * `layer.parameter()`에는 `required_grad=True` 로 지정된 변수들만 포함된다.
  * 대부분의 layer에 weights 값들이 지정되어 있어서 직접 지정할 일은 거의 없긴 함
* Backward from the scratch
  * `nn.Module`에서 `backward`와 `optimizer` 오버라이딩하면 된다.
* 추가자료
  * [Pytorch로 Linear Regression하기](https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817)
  * [Pytorch로 Logistic Regession하기](https://medium.com/dair-ai/implementing-a-logistic-regression-model-from-scratch-with-pytorch-24ea062cd856)

## [5강] Dataset & Dataloader

* `Dataset(Data, transforms)` : 데이터를 입력하는 방식의 표준화
  * `__init__()` : 초기 데이터 생성 방법 지정
  * `__len__()` : 데이터의 전체 길이
  * `__getitem__()` : index값을 주었을 때 반환되는 데이터의 형태
* `DataLoader(Dataset, batch, shuffle, ...)` : 데이터의 batch를 생성해주는 클래스
* [TORCHVISION.DATASETS](https://pytorch.org/vision/stable/datasets.html)
  * [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) : 위의 소스코드 참고해서 데이터셋 만드는 연습해보기
* 추가자료
  * [Pytorch Dataset, Dataloader 튜토리얼](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

## [6강] 모델 불러오기

* 모델의 파라미터 저장 및 로드
  * `model.state_dict()` : 모델의 파라미터 표시
  ```python
  torch.save(model.state_dict(),"model.pt") # 저장
  new_model = ModelClass()
  new_model.load_state_dict(torch.load("model.pt")) # 로드
  ```
* 모델 형태(architecture)와 파라미터 저장 및 로드
  ```python
  torch.save(model, "model.pt") # 저장
  new_model = torch.load("model.pt") # 로드
  ```
* 모델 구조 출력
  ```python
  from torchsummary import summary
  summary(model, (3, 224, 224))
  ```
  ```python
  for name layer in model.named_modules():
      print(name, layer)
  ```

* checkpoints
  * 학습의 중간 결과 저장, 일반적으로 epoch, loss, mertic을 함께 저장
    ```python
    torch.save({
        'epoch':e, 'loss': epoch_loss, 'optimizer_state_dict': optimizer.state_dict(),
        'model_statae_dict': model.state_dict()
    }, PATH) # 저장
    checkpoint = torch.load(PATH) # 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    ```

* pretrained model transfer learning
  * 다른 데이터셋(일반적으로 대용량 데이터셋)으로 만든 모델을 현재 데이터에 적용
    * 모델 추천 : [CV models](https://github.com/rwightman/pytorch-image-models) & [NLP models](https://huggingface.co/models)
  * 마지막 레이어 수정하기
    * `vgg.fc = torch.nn.Linear(1000, 1)` : 맨 마지막에 fc 레이어 추가하기 (추천)
    * `vgg.classifier._modules['6'] = torch.nn.Linear(4096, 1)` : 맨 마지막 레이어 교체하기
  * Freezing : pretrained model 활용 시 모델의 일부분을 frozen 시킨다.
    ```python
    for param in mymodel.parameters():
        param.requires_grad = False # frozen
    for param in mymodel.linear_layers.parameters():
        param.requires_grad = True # 마지막 레이어 살리기
    ```

## [7강] Monitoring tools for PyTorch

* 목표 : print문은 이제 그만!
* ✨[Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)✨ : TensorFlow의 프로젝트로 만들어진 시각화 도구, PyTorch도 연결 가능
  * 종류
    * scalar : metric 등 표시
    * graph : 계산 그래프 표시
    * histogram : 가중치 등 값의 분포 표시
    * image / text : 예측값과 실제값 비교
    * mesh : 3d 형태로 데이터 표형 (위에 비해 자주 쓰진 않음)
  * 방법
    * 기록을 위한 디렉토리 생성 : `logs/[실험폴더]` 로 하는 것이 일반적
    * 기록 생성 객체 `SummaryWriter` 생성
    * `writer.add_scalar()` 등으로 값들을 기록
    * `writer.flush()` : disk에 쓰기
    * `%load_ext tensorboard` : 텐서보드 부르기
    * `%tensorboard --logdir {logs_base_dir}` : 6006포트에 자동으로 텐서보드 생성
* ✨[Weight & Biases(WandB)](https://wandb.ai/site)✨ : 머신러닝 실험 지원, MLOps의 대표적인 툴이 되고 있다.
  * 기능
    * 협업 / code versioning / 실험 결과 기록 등
    * 유료지만, 무료 기능도 있다.
  * 방법
    * 홈페이지 : 회원가입 -> API 키 확인 -> 새 프로젝트 생성
    * `wandb.init(project, entity)` : 여기서 API 입력해서 접속
    * `wandb.init(project, config)` : config 설정
    * `wandb.log()` : 기록
* [Pytorch Lightning Logger 목록](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)

## [8강] Multi-GPU 학습

* 정의
  * Multi-GPU : GPU를 2개 이상 쓸 때
  * Single Node Multi GPU : 한 대의 컴퓨터에 여러 개의 GPU
* 방법
  * 모델 병렬화(Model parallel)
    * 모델을 나누기 (ex. AlexNet)
    * 모델의 병목, 파이프라인 어려움 등의 문제
      ```python
      # __init__()
      self.seq1 = nn.Sequential(~~).to('cuda:0') # 첫번째 모델을 cuda 0에 할당
      self.seq2 = nn.Sequential(~~).to('cuda:1') # 두번째 모델을 cuda 1에 할당
      # forward()
      x = self.seq2(self.seq1(x).to('cuda:1')) # 두 모델 연결하기
      ```
  * 데이터 병렬화(Data parallel)
    * 데이터를 나눠 GPU에 할당한 후, 결과의 평균을 취하는 방법
    * `DataParallel`
      * 특징 : 단순히 데이터를 분배한 후 평균 (중앙 코디네이터 필요)
      * 문제 : GPU 사용 불균형, Batch 사이즈 감소, GIL(Global Interpreter Lock)
      * `parallel_model = torch.nn.DataParallel(model)`
    * `DistributedDataParallel` : 각 CPU마다 process 생성하여 개별 GPU에 할당
      * 특징 : 개별적으로 연산의 평균을 냄 (중앙 코디네이터 불필요, 각각이 코디네이터 역할 수행)
      * 방법 : 각 CPU마다 process 생성하여 개별 GPU에 할당 (CPU도 GPU 개수만큼 할당)
        ```python
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, 
            shuffle=False, pin_memory=True, num_workers=[GPU개수x4],
            sampler=train_sampler # sampler 사용
        )
        ...
        # Distributed dataparallel 정의
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        ```
      * 더 자세한 코드는 [여기](https://blog.si-analytics.ai/12)
* 추가자료
  * [PyTorch Lightning - MULTI-GPU TRAINING](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html)
  * [PyTorch - GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  * [관련 집현전 영상](https://youtu.be/w4a-ARCEiqU?t=1978)
  * TensorRT : NVIDIA가 제공하는 도구

## [9강] Hyperparameter Tuning

* 목표 : 마지막 0.01의 성능이라도 높여야 할 때!
* 기법 : Grid Search / Random Search / 베이지안 기법(BOHB 등)
* ✨[Ray](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)✨ : multi-node multi processing 지원 모듈, Hyperparameter Search를 위한 다양한 모듈 제공
  * 방법
    * `from ray import tune`
    * config에 search space 지정
      ```python
      config = {
              "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
              "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
              "lr": tune.loguniform(1e-4, 1e-1),
              "batch_size": tune.choice([2, 4, 8, 16])}
      ```
    * 학습 스케줄링 알고리즘 지정
      ```python
      scheduler = ASHAScheduler(
              metric="loss",
              mode="min",
              max_t=max_num_epochs,
              grace_period=1,
              reduction_factor=2)
      ```
    * 결과 출력 양식 지정
      ```python
      reporter = CLIReporter(
              # parameter_columns=["l1", "l2", "lr", "batch_size"],
              metric_columns=["loss", "accuracy", "training_iteration"])
      ```
    * 병렬 처리 양식으로 학습 실행
      ```python
      result = tune.run(
              partial(train_cifar, data_dir=data_dir), # train_cifar : full training function
              resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
              config=config,
              num_samples=num_samples,
              scheduler=scheduler,
              progress_reporter=reporter)
      ```
    * 학습 결과 가져오기
      * `best_trial = result.get_best_trial("loss", "min", "last")`

## [10강] Pytorch Troubleshooting

* OOM(Out Of Memory) : 왜, 어디서 발생했는지 알기 어려움
* 쉬운 방법 : Batch size 줄이고 GPU clean 한 후 다시 실행
* 다른 방법
  * `GPUtil.showUtilization()` : GPU의 상태를 보여준다.
  * `torch.cuda.empty_cache()` : 사용되지 않은 GPU상 cache 정리 (학습 loop 전에 실행하면 좋다)
  * `total_loss += loss.item` : `total_loss += loss` 에서는 계산 그래프를 그릴 필요가 없기 때문에,  python 기본 객체로 변환하여 더해준다.
  * `del 변수` : 필요가 없어진 변수는 적절히 삭제하기 (정확히는 변수와 메모리 관계 끊기)
  * try-except문을 이용해서 가능한 batch size 실험해보기
  * `with torch.no_grad()` : Inference 시점에 사용
  * tensor의 float precision을 16bit로 줄일 수도 있다. (많이 쓰이진 않음)
* [이 외 GPU 에러 정리 블로그](https://brstar96.github.io/shoveling/device_error_summary/)
