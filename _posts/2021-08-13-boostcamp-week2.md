---
layout: post
title:  "[Boostcamp AI Tech] 2주차 - DL Basic"
subtitle:   "Naver Boostcamp AI Tech Level1 U Stage"
categories: "Boostcamp-AI-Tech"
tags: [Level1-U-Stage]
use_math: true
---

# 부스트캠프 2주차

## [1강] 딥러닝 기본 용어 설명

* 딥러닝 중요 요소 : data, model, loss function, optimization algorithm 등
  * loss function : 이루고자 하는 것의 근사치
    * ![image](https://user-images.githubusercontent.com/35680202/128652035-6e7f92a1-2929-4e2f-9b05-0873aefdc5d8.png)
* [Historical Review](https://dennybritz.com/blog/deep-learning-most-important-ideas/)
  * 2012 - AlexNet : ImageNet challenge 에서 딥러닝 기법으로 처음 1등
  * 2013 - DQN : 강화학습 Q러닝, 딥마인드
  * 2014 - Encoder/Decoder : NMT(Neural Machine Translation) 기계어번역
  * 2014 - Adam Optimizer : 웬만하면 학습이 잘 된다.
  * 2015 - GAN : 네트워크가 generator와 discriminator 두개를 만들어서 학습
  * 2015 - Residual Networks : 네트워크를 깊게 쌓을 수 있게 만들어줌
  * 2017 - Transformer (Attention Is All You Need) : 기존 방법론들을 대체할 정도의 영향력
  * 2018 - BERT (Bidirectional Encoder Representations from Transformers) : 'fine-tuned' NLP models 발전 시작
  * 2019 - BIG Language Models : fine-tuned NLP model의 끝판왕, OpenAI GPT-3
  * 2020 - Self-Supervised Learning : SimCLR (a simple framework for contrastive learning of visual representations)

## [2강] 뉴럴 네트워크

* Neural networks are **function approximators** that stack affine tansformations  followed by nonlinear transformations : 행렬 곱과 비선형 연산이 반복되면서, 함수를 근사하는 모델
* Linear Neural Networks
  * $y = W^Tx + b$에서 $W$를 찾는 것은 서로 다른 두 차원에서의 선형변환을 찾겠다는 것
* Multi-Layer Perceptron
  * $y = W_2^Th = W_2^T W_1^T x$​ 는 linear neural network와 다를 바가 없다.
  * $y = W_2^Th = W_2^T \rho(W_1^T x)$​와 같은 Nonlinear transform이 필요하다.
  * Multilayer Feedforward Networks are Universal Approximators : 뉴럴 네트워크의 표현력이 그만큼 크다. 하지만 어떻게 찾을지는 (알아서해)
* [PyTorch official docs](https://pytorch.org/docs/stable/nn.html)

## [3강] Optimization

* **Generalization** : 일반화 성능을 높이는 것이 우리의 목표
  * Generalization gap : Training error와 Test error의 차이
  * **(k-fold) Cross-validation** : 최적의 하이퍼파라미터 찾기
* **Bias and Variance Tradeoff**
  * Bias : 평균적으로 봤을 때 정답에 가까우면 bias가 낮음
  * Variance : 출력이 일관적이면 variance가 낮음
  * Tradeoff
    * minimizing cost = (bias^2, variance, noise) 를 낮추는 것
    * 하나가 낮으면 하나가 높을 수 밖에 없다
    ![image](https://user-images.githubusercontent.com/35680202/128812048-311e7b47-9543-4c0c-b5ce-1b7e2ba9e725.png)

* **Bootstrapping** : test하거나 metric을 계산하기 전에 random sampling하는 것
  * **Bagging**(Bootstrapping aggregating)
    * bootstrapping을 이용해서 여러 모델을 학습시킨 후 결과를 합치겠다.(voting or averaging)
    * 모든 모델이 독립적으로 돌아감
  * **Boosting**
    * 하나하나의 모델들을 시퀀셜하게 합쳐서 하나의 모델을 만든다.
    * 이전 모델이 잘 예측하지 못한 부분을 보완하기 위한 방식으로 학습해나감

* **Gradient Descent** : 1차 미분 이용, local minimum을 찾는 알고리즘
  * Stochastic gradient descent : 엄밀히 말하면 SGD는 한개의 샘플로 업데이트 하는 것
  * Mini-batch gradient descent
  * Batch gradient descent

* Batch-size Matters : 올바른 배치 사이즈는?
  * 배치 사이즈를 작게 쓰면 Flat minimizer에 수렴 : generalization performance가 더 높다.
  * 배치 사이즈를 크게 쓰면 Sharp minimizer에 수렴
    ![image](https://user-images.githubusercontent.com/35680202/128814469-2ec789a8-e130-4c2b-8cfa-744e1e990685.png)

* Gradient Descent Methods
  * **(Stochastic) gradient descent** : 적절한 learning rate를 넣는 것이 중요
    * $W_{t+1} \leftarrow W_t - \eta g_t$​
      * $\eta$ : Learning rate
      * $g_t$ : Gradient
  * **Momentum** : 관성, 현재 gradient를 가지고 momentum을 accumulation 한다. 한번 흘러간 gradient direction을 어느정도 유지시켜주기 때문에 gradient가 왔다갔다해도 어느정도 잘 학습된다.
    * $a_{t+1} \leftarrow \beta a_t + g_t$
      * $\beta$ : momentum
      * $a_{t+1}$ : accumulation
    * $W_{t+1} \leftarrow W_t - \eta a_{t+1}$
  * **Nesterov Accelerated Gradient** : 현재 방향으로 한번 가보고 그곳에서 gradient를 구한걸 가지고 accumulation 한다. 
    * $a_{t+1} \leftarrow \beta a_t + \nabla L(W_t - \eta \beta a_t)$​
      * $\nabla L(W_t - \eta \beta a_t)$ : Lookahead gradient
      * $a_{t+1}$ : accumulation
    * $W_{t+1} \leftarrow W_t - \eta a_{t+1}$
  * **Adagrad** : (Adaptive) 파라미터가 지금까지 얼마나 변해왔는지 아닌지를 보고, 많이 변한 파라미터는 적게 변화시키고, 안 변한 파라미터는 많이 변화시키고 싶은 것
    * $W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$​​
      * $G_t$ : Sum of gradient squares, 지금까지 gradient가 얼마나 많이 변했는지를 제곱해서 더한 것, 학습 중에 계속 커지기 때문에 뒤로 갈수록 학습이 멈출 수도 있음
      * $\epsilon$ : for numerical stability
  * **Adadelta** : Adagrad에서 learning rate이 $G_t$​의 역수로 표현됨으로써 생기는 monotonic한 decreasing property를 막는 방법, no learning rate, 사실 많이 사용되지는 않음
    * $G_t = \gamma G_{t-1} + (1 - \gamma) g_t^2$​ : EMA(exponential moving average) of gradient squares
    * $W_{t+1} = W_t - \frac{\sqrt{H_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} g_t$
    * $H_t = \gamma H_{t-1} + (1 - \gamma) (\Delta W_t)^2$​​ : EMA of difference squares
  * **RMSprop** : Geoff Hinton의 강의에서 제안됨
    * $G_t = \gamma G_{t-1} + (1 - \gamma) g_t^2$​ : EMA of gradient squares
    * $W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$​
      * $\eta$ : stepsize
  * **Adam** : Adaptive Moment Estimation, 무난하게 사용하는 방법, adaptive learning rate approach와 Momentum 두 가지 방식을 결합한 것
    * $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$​ : Momentum
    * $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ : EMA of gradient squares
    * $W_{t+1} = W_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} m_t$​
      * $\eta$ : stepsize
      * $\epsilon$ : 실제로 적용할 때, 입실론 값을 잘 조절하는 것이 중요하다.
  * [**RAdam**](https://github.com/LiyuanLucasLiu/RAdam)
  * [**AdamP**](https://github.com/clovaai/AdamP)

* Regularization
  * Early stopping : 오버피팅 전에 학습 종료하기
  * Parameter norm penalty : 부드러운(smoothness) 함수를 만들기 위함
  * Data augmentation : 주어진 데이터셋을 어떻게든 늘려서 사용하는 것
  * Noise robustness : 입력 데이터 또는 가중치에 노이즈를 넣는 것
  * Label Smoothing : mix-up, cutmix 등을 통해 decision boundary를 부드럽게 만드는 것
  * Dropout : 랜덤하게 가중치를 0으로 만드는 것, robust한 feature를 잡을 수 있기를 기대
  * Batch Normalization : 정규화하고자 하는 레이어의 statistics를 정규화하는 것

## [4강] Convolution

* 2D Image Convolution
  * $(I * K)(i, j) = \sum_m \sum_n I(m,n) K(i-m, j-n) = \sum_m \sum_n I(i-m, i-n) K(m,n)$
* RGB Image Convolution
  * ![image-20210811103921000](https://user-images.githubusercontent.com/35680202/128956799-093d9612-2e44-455e-bbe3-7bfb986e7d70.png)
* Stack of Convolutions
  * Conv -> ReLU -> Conv -> ReLU
* Convolutional Neural Networks
  * feature extraction : conv layer, pooling layer
  * decision making : fully connected layer
* 1x1 Convolution
  * 목표 : Dimension(채널) reduction - 파라미터 수 줄이기
  * ex) bottleneck architecture

## [5강] Modern CNN

* ILSVRC(ImageNet Large-Scale Visual Recognition Challenge)
  * 분류(Classification) : 1000개의 카테고리
  * 데이터셋 : 100만 장 이상, 학습 데이터셋 : 45만 장
  * Human 성능 : 5.1% <- 2015년도부터 사람의 성능을 따라잡기 시작
* **AlexNet**
  * 구조
    * 5 conv layers + 3 dense layers = 8 layers
    * 11x11 filters : 파라미터가 많이 필요하다.
  * 핵심
    * ReLU(Rectified Linear Unit) activation 사용 : vanishing gradient problem 해결
    * 2개의 GPU 사용
    * Data augmentation, Dropout
  * 의의 : 일반적으로 제일 잘 되는 기준을 잡아준 모델
* **VGGNet**
  * 구조
    * **3x3 filters 만 사용**
    * 1x1 conv 를 fully connected layers 에서 사용
    * 레이어 개수에 따라서 VGG16, VGG19
    * Dropout (p=0.5)
  * 핵심
    * 3x3 conv 두 번이면, 5x5 conv 한 번과 receptive field가 (5x5)로 같다.
      * 3x3 conv 두 개의 파라미터 수 : (3x3x128x128) x2 = 294,912
      * 5x5 conv 한 개의 파라미터 수 : (5x5x128x128) x1 = 409,600
* **GoogLeNet**
  * 구조
    * 22 layers
    * network in network : 비슷한 네트워크가 네트워크 안에서 반복됨
    * Inception blocks
      * ![image](https://user-images.githubusercontent.com/35680202/128962913-0ce88766-ceb8-4511-ab3a-d8063d161620.png)
      * 여러 개의 receptive field를 가지는 filter를 거치고, 결과를 concat 하는 효과
      * 1x1 conv 가 끼워지면서 전체적인 파라미터 수를 줄일 수 있게 된다. (channel-wise dimension reduction)
  * 핵심
    * **1x1 conv 의 이점**
      * (1) in_c=128, out_c=128, **3x3 conv** 의 파라미터 수 : (3x3x128x128) = 147,456
      * (2-1) in_c=128, out_c=32, **1x1 conv** 의 파라미터 수 : (1x1x128x32) = 4096
      * (2-2) in_c=32, out_c=128, **3x3 conv** 의 파라미터 수 : (3x3x32x128) = 36,864
      * (1) >>> (2-1)+(2-2)
* **ResNet**
  * 문제 : 깊은 네트워크가 학습시키기 어렵다. 오버피팅 아니고, 학습이 잘 안 되는 것
  * 구조
    * Identity map (**skip connection**, residual connection)
      * ![image](https://user-images.githubusercontent.com/35680202/128962878-0a05ee59-c227-4eb4-837b-8a2393228ab0.png)
    * Shortcut
      * Simple Shortcut : 입력 x 그대로 사용(차원이 같은 경우)
      * Projected Shortcut : x 에 1x1 conv를 통과시켜 channel depth를 match 시킨다.
    * Bottleneck architecture
      * ![image](https://user-images.githubusercontent.com/35680202/128963667-aecf7b9c-f8d4-449c-bd21-6c3d29c37b39.png)
      * 3x3 conv 하기 전/후에 1x1 conv 로 채널 수를 줄이고, 다시 늘리는 방법
  * 의의
    * 네트워크를 더 깊게 쌓을 수 있는 가능성을 열어줌
    * Performance는 증가하는데 parameter size는 감소하는 방향으로 발전
* **DenseNet**
  * 구조
    * Dense Block
      * ![image](https://user-images.githubusercontent.com/35680202/128963758-d6d7ea6e-0651-493a-aaf7-39940e3706f6.png)
      * addition 대신 **concatnation** 을 사용한다.
      * 채널이 기하급수적으로 커지게 된다.
    * Transition Block (for Dimension reduction)
      * batchnorm -> 1x1 conv -> 2x2 avgpool
  * 웬만하면 resnet이나 densenet을 쓰면 성능이 좋다.

## [6강] Computer Vision Applications

* **Semantic Segmentation**
  * 이미지의 모든 픽셀이 어떤 라벨에 속하는지 보고 싶은 것
  * 자율주행에 가장 많이 활용됨
  * **fully convolutional network**
    * convolutionalization
      * dense layer를 없애고 싶음
      * 파라미터의 수는 똑같다.
      * ![image](https://user-images.githubusercontent.com/35680202/128982583-c7d609f2-4b7d-465b-b058-147f4d07f547.png)
    * 특징
      * Input 의 spatial dimension 에 독립적이다.
      * heatmap 같은 효과가 있다.
    * Deconvolution (conv transpose)
      * dimension이 줄어들기 때문에 upsample 이 필요하다.
      * convolution의 역 연산이라고 생각하면 편함
      * spatial dimension을 키워준다.
* **Detection**
  * 어느 객체가 어디에 있는지 bounding box 를 찾고 싶은 것
  * R-CNN
    * 방법
      * 이미지 안에서 region을 여러 개 뽑는다. (Selective search)
      * 똑같은 크기로 맞춘다.
      * 피쳐를 뽑아낸다. (AlexNet)
      * 분류를 진행한다. (SVM)
    * 문제
      * region을 뽑은 만큼 CNN에 넣어서 계산해야하니까 계산량도 많고 오래 걸린다.
  * SPPNet
    * 목표 : 이미지를 CNN에서 한번만 돌리자
    * 방법
      * 이미지 안에서 bounding box를 뽑고, 
      * 이미지 전체에 대해서 convolutional feature map을 만든 다음에, 
      * 뽑힌 bounding box에 해당하는 convolutional feature map의 텐서만 가져와서 쓰자
      * 결론적으로, CNN을 한번만 돌아도 텐서를 뜯어오는 것만 region별로 하기 때문에 훨씬 빨라진다.
    * 한계
      * 여전히 여러 개의 region을 가져와서 분류하는 작업이 필요
  * Fast R-CNN
    * 방법
      * SPPNet과 거의 동일한 컨셉
      * 뒷단에 neural network와 RoI feature vector를 통해서 bounding box regression과 classification을 했다는 점
  * Faster R-CNN
    * 목표 : bounding box를 뽑는 것도 network로 학습하자
    * 방법 : **Region Proposal Network(RPN)** + Fast R-CNN
      * Region Proposal Network(RPN) : 이미지의 특정 영역(패치)가 bounding box로서의 의미가 있을지 없을지를 찾아준다. 물체가 무엇인지는 뒷단에 있는 네트워크가 해줌
        * Anchor boxes : 미리 정해놓은 bounding box의 크기 (대충 이 이미지에 어떤 크기의 물체가 있을 것 같은지 정함, 템플릿같은 것)
        * ![image](https://user-images.githubusercontent.com/35680202/129042366-6e6a755c-476b-492f-bfa8-dd962f90dd58.png)
  * YOLO (v1)
    * 목표 : No explicit bounding box sampling
      * You Only Look Once
      * Faster R-CNN 보다 훨씬 빠르다.
    * 방법
      * 이미지가 들어오면 SxS grid 로 나눈다.
      * 찾고 싶은 물체의 중앙이 해당 grid 안에 들어가면 그 grid cell이 해당 물체에 대한 bounding box와 그 해당 물체가 무엇인지를 같이 예측
      * 각각의 cell은 B개의 bounding box 예측 + C개의 class에 대한 probabilities
    * 정리
      * SxS x (B*5 + C) tensor
        * SxS : Number of cells of the grid
        * B*5 : B bounding boxes with offsets (x,y,w,h) and confidence(필요성)
        * C : Number of classes

## [7강] Sequential Models - RNN

* Sequential Data
  * 오디오, 비디오 등
  * 입력의 차원을 알 수 없어서 처리하는데에 어려움
  * 몇 개의 입력이 들어오는지에 상관없이 모델은 동작해야 함
* Naive Sequential Model
  * 이전의 입력에 대해서 다음에 어떤 출력이 나올지 예측
  * $p(x_t \| x_{t-1}, x_{t-2}, ...)$​​
    * $x_{t-1}, x_{t-2}, ...$​ : The number of inputs varies, 고려해야하는 과거의 정보가 점점 늘어남
  * $p(x_t \| x_{t-1}, ..., x_{t-r})$
    * $x_{t-r}$ : Fix the past timespan, 과거의 r 개의 정보만 고려한다.
* Markov model (first-order autogressive model)
  * $p(x_1,...,x_T) = p(x_T \| x_{T-1}) p(x_{T-1} \| x_{T-2}) ... p(x_2 \| x_1) p(x_1) = \Pi_{t=1}^{T} p(x_t \| x_{t-1})$
  * 현재는 (바로 전) 과거에만 의존적이다.
  * 과거의 많은 정보를 버리는 것이 됨
* Latent autogressive model 
  * $\hat{x} = p(x_t \| h_t)$
    * $h_t = g(h_{t-1}, x_{t-1})$
    * $h_{t-1}, h_t$ : summary of the past
  * 현재는 바로 전 과거 하나가 아니라, 이전의 정보를 요약하는 hidden state에 의존적이다.
* RNN(Recurrent Neural Network)
  * 앞서 나온 내용을 가장 쉽게 구현하는 방법
  * ![image](https://user-images.githubusercontent.com/35680202/129124658-70742af7-2c9d-4881-8e3e-c97e643a1a0d.png)
  * 단점
    * Long-term dependencies를 잡는 것이 어렵다.
      * 먼 과거에 있는 정보가 미래에 영향을 주기까지 남아있기가 어렵다.
    * 학습이 어렵다.
      * $h_1 = \Phi(W^T h_0 + U^T x_1)$
      * ...
      * $h4 = \Phi(W^T \Phi(W^T \Phi(W^T \Phi(W^T h_0 + U^T x_1) + U^T x_2) + U^T x_3) + U^T x_4)$
        * ex) activation function이 sigmoid인 경우 : vanishing gradient
        * ex) activation function이 relu인 경우 : exploding gradient
* LSTM(Long Short Term Memory)
  * 구조
    * Forget gate : Decide which information to throw away
      * $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
    * Input gate : Decide which information to store in the cell state
      * $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
    * Update cell
      * $\tilde{C_t} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ : 예비 cell state
      * $C_t = f_t * C_{t-1} + i_t * \tilde{C_t}$ : cell state (timestep t 까지 들어온 정보 요약)
    * Output gate : Make output using the updated cell state
      * $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
      * $h_t = o_t * \tanh(C_t)$
* GRU(Gated Recurrent Unit)
  * 구조
    * Update gate
      * $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$​
    * Reset gate
      * $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$​
    * Update hidden state
      * $\tilde{h_t} = \tanh(W \cdot [r_t * h_{t-1}, x_t])$
      * $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h_t}$
  * 핵심
    * No cell state, just hidden state

## [8강] Sequential Models - Transformer

* **Transformer** is the first sequence transduction model based entirely on **attention**.
* Sequence to Sequence : NMT(Neural Machine Translation) 등
  * 입력시퀀스와 출력시퀀스의 단어의 개수가 다를 수 있다.
  * 입력시퀀스의 도메인과 출력시퀀스의 도메인이 다를 수 있다.
  * 근데 모델은 하나의 모델로 학습해야 함
* Transformer
  * **Encoder - Decoder** 구조 핵심
    * n개의 단어가 어떻게 Encoder에서 한번에 처리되는가
    * Encoder와 Decoder 사이에 어떤 정보가 주고 받아지는지
    * Decoder가 어떻게 generation할 수 있는지
  * 특징
    * CNN, MLP와 달리, 입력이 고정되더라도 옆에 주어진 다른 입력들이 달라짐에 따라서 출력이 달라질 수 있는 여지가 있다.
    * 따라서 훨씬 더 flexable하고 더 많은 것을 표현할 수 있다.
    * 하지만 계산량은 한번에 계산하기 때문에, (입력의 길이)^2이다.
* Encoder : **Self-Attention** + Feed-forward Neural Network
  * Feed-forward Neural Network : word-independent and parallelized
  * **Self-Attention**
    * 각 단어의 임베딩 벡터를 3가지 벡터(Queries(Q), Keys(K), Values(V))로 encoding 한다. - 몇 차원으로 할지는 hyperparameter
    * x1이 z1으로 변환될 때, 단순히 x1만 보는 것이 아니라 x2, x3도 같이 본다.
    * 따라서 Self-Attention은 dependencies가 있다.
  * 인코딩 과정
    * ![image](https://user-images.githubusercontent.com/35680202/129227050-b686786e-9872-4628-bcf4-ed656f6b823b.png)
    * Embedding vector : 단어를 임베딩한 벡터
    * **Query vector, Key vector, Value vector** : 각각의 neural network를 통해서 두 벡터를 만든다. (Q와 K는 내적해야하기 때문에 항상 차원이 같아야 한다.)
      * <img src="https://user-images.githubusercontent.com/35680202/129230769-be6854a4-f289-4724-91e7-c11961352222.png" width="300" height="300">
      * 행렬을 활용하여 한번에 찾을 수 있다.
      * $X$​​​ : 두 단어에 대한 4차원 임베딩 벡터
      * $W^Q, W^K, W^V$를 찾는 multi-layer perceptron이 있다.
      * 이 multi-layer perceptron은 인코딩되는 단어들마다 다 shared 된다.
    * Score vector : i번째 단어에 대한 Score vector를 계산할 때, i번째 단어의 Query vector와 자기 자신을 포함한 나머지 단어들의 Key vector들을 내적한다.
      * 두 벡터가 얼마나 align이 잘 되어있는지 본다.
      * i번째 단어가 자기 자신을 포함한 나머지 단어들과 얼마나 관계가 있는지를 나타냄 (이것이 결국 attention)
      * 그 후 Score vector를 normalize해주고 softmax를 취해준다.
    * 최종 결과물 : Value vector 와 weighted sum을 해서 만든 인코딩 된 벡터
      * <img src="https://user-images.githubusercontent.com/35680202/129231629-e6b8b9e2-f8c9-4ade-9d77-971e08f2e7c6.png" width="450" height="200">
      * Value vector의 weight를 구하는 과정이 각 단어에서 Query vector와 Key vector 사이의 내적을 normalize&softmax 취해주고 나오는 attention을 Value vector와 weighted sum을 한 것이 최종적으로 나오는 인코딩된 벡터
      * 여기서는 인코딩된 벡터가 Value vector의 차원과 같다.
  * **Multi-headed attention(MHA)**
    * attention을 여러 번 하는 것
    * 즉, 하나의 임베딩된 벡터에 대해서 Query, Key, Value를 하나만 만드는 것이 아니라 N개 만드는 것
    * 따라서 하나의 임베딩된 벡터가 있으면 N개의 인코딩된 벡터를 얻을 수 있다.
    * 그 다음 인코더에 넣을 때 차원을 맞춰주기 위해서 다시 행렬을 곱해준다.
  * 최종 그림
    * ![image](https://user-images.githubusercontent.com/35680202/129234393-0f693f3d-e621-4462-9bd4-230ecdcb169c.png)
    * 사실 실제 구현이 이렇지는 않다.
    * Embedding dimension이 100이고, 10개의 head를 사용한다고 하면, 100 dimension을 10개로 나눠서 각각 10 dimension짜리를 가지고 Q, K, V를 만든다.
  * Positional encoding
    * 입력에 특정 값을 더해주는 것 (bias라고 보면 됨)
    * Transformer구조가 N개의 단어의 Sequential한 정보가 사실 포함되어 있지 않기 때문에 사용함
    * Positional encoding은 pre-defined된 특정 방법으로 벡터를 만든다.
* Decoder
  * Self-Attention
    * Decoder의 self-attention layer 에서 이전 단어들만 dependent하고 뒤(미래)에 있는 단어들은 dependent하지 않게(활용하지 않게) 만들기 위해 마스킹한다.
  * Encoder-Decoder Attention
    * Encoder에서 Decoder로 Key vector와 Value vector를 보낸다.
    * 이전 레이어의 Query vector와 Encoder에서 받아온 Key vector와 Value vector들을 가지고 최종 값이 나오게 된다.
  * Final layer
    * 단어들의 분포를 만들어서 그 중의 단어 하나를 매번 sampling 하는 식으로 동작
    * 출력은 Autoregressive 하게 하나의 단어씩 만든다. (I 가 들어가면 am 이 출력되는)
* Vision Transformer(ViT)
  * 이미지 분류를 할 때 Transformer Encoder만 활용한다.
  * Encoder에서 나온 첫번째 encoded vector를 classifier에 집어넣는 방식
  * 이미지에 맞게 하기 위해서 이미지를 영역으로 나누고 서브패치들을 linear layer를 통과해서 그게 하나의 입력인 것처럼 넣어준다. (positional embedding 포함)
* [DALL-E](https://openai.com/blog/dall-e/)
  * 문장에 대한 이미지를 만들어낸다.

## [9강] Generative Models 1

* **Generative model** : 생성 모델, probability distribution $p(x)$ 를 배우는 것이다.
  * 기능
    * Generation : $x_{new} \sim p(x)$​​ 을 sampling하면, 마치 강아지같은 이미지를 얻을 수 있다.
    * Density estimation : $p(x)$ 로 $x$​가 강아지와 비슷한지 아닌지를 구분할 수 있다. (anomaly detection에 사용될 수 있다.)
    * Unsupervised representation learning : 강아지 이미지에는 보통 귀, 꼬리가 있다는 특성을 배우는 것 (feature learning)
  * 종류
    * explicit (generative) model : 입력이 주어졌을 때, 이 입력에 대한 확률값을 얻어낼 수 있는 모델
    * implicit (generative) model : 단순히 generation만 하는 모델
  * 핵심 : $p(x)$​를 어떻게 만들(표현할) 것인가?
* Basic Discrete Distributions : 관심있어하는 값들이 finite set인 경우
  * Bernoulli distribution : (biased) coin flip, 0 또는 1(head or tail)이 나옴
    * $D = [Heads, Tails]$
    * $P(X = Heads) = p$ 라고 하면, $P(X = Tails) = 1 - p$ 가 된다.
    * $X \sim Ber(p)$​ 라고 표기한다.
    * 확률을 표현하는데 한 개의 파라미터가 필요하다.
  * Categorical distribution : (biased) m-sided dice
    * $D = [1, ..., m]$
    * $P(Y = i) = p_i$ 라고 하면, $\sum_{i=1}^{m} p_i = 1$ 이다.
    * $Y \sim Cat(p_1, ..., p_m)$​ 라고 표기한다.
    * 확률을 표현하는데 m-1개의 파라미터가 필요하다.
* 예제
  1. (한 픽셀에 대한) RGB joint distribution 을 만들어보자
     * $(r, g, b) \sim p(R, G, B)$
     * 경우의 수 : 256 x 256 x 256
     * 확률을 표현하는데 필요한 파라미터 수 : (256 x 256 x 256) - 1
     * 즉, 하나의 RGB픽셀을 fully discribe하기 위해서 필요한 파라미터의 숫자가 엄청 크다.
  2. n개의 binary pixels ($X_1, ..., X_n$)를 가지는 binary image 하나가 있다고 하자
     * 경우의 수 : $2^n$
     * 확률 $p(x_1,...,x_n)$을 표현하는데 필요한 파라미터 수 : $2^n - 1$​
* **Structure Through Independence**
  * 동기 : 기계학습에서 파라미터 수가 늘어나면 학습은 더 어렵다.
  * 만약 위 예제 2번에서 $X_1, ..., X_n$​개의 픽셀들이 모두 independent 하다고 생각하면 어떨까? (말이 안 되는 가정이긴 함)
    * $p(x_1, ..., x_n) = p(x_1) p(x_2) ... p(x_n)$ 으로 나타낼 수 있다.
    * 가능한 경우의 수 : $2^n$ (위와 똑같다.)
    * 확률 $p(x_1,...,x_n)$을 표현하는데 필요한 파라미터 수 : $n$
* **Conditional Independence**
  * 동기
    * 픽셀들이 fully dependent 하면 너무 많은 파라미터가 필요하다.
    * 픽셀들이 모두 independent 하면 파라미터는 줄어들어서 좋은데, 표현할 수 있는 이미지가 너무 적다. (일반적으로 우리가 아는 이미지를 전혀 만들 수 없다.)
    * 이 중간 어딘가 적절한 것을 찾고 싶다.
  * 핵심
    * Chain rule : n개의 joint distribution을 n개의 conditional distribution으로 표현해주는 것 (independent한 것과 관련 없이 항상 만족한다.)
      * $p(x_1, ..., x_n) = p(x_1) p(x_2 \| x_1) p(x_3 \| x_1, x_2) ... p(x_n \| x_1, ..., x_{n-1})$
    *  Bayes' rule : 
      * $p(x \| y) = \frac{p(x, y)}{p(y)} = \frac{p(y \| x) p(x)}{p(y)}$
    * Conditional independence 
      * 만약 $x \perp y  \|  z$ 이면, $p(x \| y,z) = p(x \| z)$ 이다.
      * z가 주어졌을 때, x와 y가 independent하다면(conditional independent), x를 표현하는데에 z가 주어지면 y는 상관이 없어진다.(뒷단의 conditional 부분을 날려줄 수 있다.)
  * 목표 : Conditional independence와 Chain rule을 잘 섞어서 좋은 모델을 만들자
  * 방법
    * Chain rule 을 사용하여 표현하자
      * $p(x_1, ..., x_n) = p(x_1) p(x_2 \| x_1) p(x_3 \| x_1, x_2) ... p(x_n \| x_1, ..., x_{n-1})$
      * 확률을 표현하는데 필요한 파라미터 수?
        * $p(x_1)$ : 1개
        * $p(x_2  \|  x_1)$ : 2개 => $p(x_2 \| x_1 = 0)$, $p(x_2  \|  x_1 = 1)$
        * $p(x_3  \|  x_1, x_2)$ : 4개 => $p(x_3 \| x_1 = 0, x_2=0)$, $p(x_3 \| x_1 = 1, x_2=0)$, $p(x_3 \| x_1 = 0, x_2=1)$, $p(x_3 \| x_1 = 1, x_2=1)$
        * 즉, $2^n - 1$개 (fully independent와 똑같다)
    * Markov assumption 을 가정하자
      * i+1 번째 픽셀은 i 번째 픽셀에만 dependent 하다.
      * $X_{i+1} \perp X_1,...,X_{i-1}  \|  X_i$
    * Conditional independence 에 의해​,
      * $p(x_1, ..., x_n) = p(x_1) p(x_2 \| x_1) p(x_3 \| x_2) ... p(x_n \| x_{n-1})$​​​ 으로 바뀐다.
      * 확률을 표현하는데 필요한 파라미터 수?
        * $p(x_1)$ : 1개
        * $p(x_2  \|  x_1)$ : 2개 => $p(x_2 \| x_1 = 0)$, $p(x_2  \|  x_1 = 1)$
        * $p(x_3  \|  x_2)$ : 2개 => $p(x_3 \| x_2 = 0)$, $p(x_3  \|  x_2 = 1)$
        * 즉, $2n - 1$​개
* **Auto-regressive Model**
  * 핵심
    * 하나의 정보가 이전 정보에 dependent 한 특징을 가지는 모델들을 전반적으로 지칭
    * 위의 Conditional Independence를 잘 이용한 모델도 포함
    * 이전 정보 n개에 dependent한 모델을 ar-n 모델이라고 함
  * **Neural Autoregressive Density Estimator(NADE)**
    * 방법
      * i 번째 픽셀이 1부터 i-1 번째 픽셀에 dependent 하다고 가정
      * $p(x_i  \|  x_{1 : i-1}) = \sigma (\alpha_{i} h_i + b_i)$​ (이때, $h_i = \sigma (W_{< i} x_{1 : i-1} + c)$​)
      * 100번째 픽셀에 대한 확률분포를 만들기 위해서 99개의 이전 입력들을 받을 수 있는 neural network가 필요하다.
    * 특징
      * explicit model 이다. (입력이 들어오면 이 입력의 확률을 구할 수 있다.)
    * 출력
      * binary output이면 그냥 sigmoid 통과한다.
      * continuous output이면 마지막 레이어에 가우시안 mixture 모델을 사용해서 continuous한 distribution을 만든다.
  * **Pixel RNN**
    * 방법
      * $n \times n$​ RGB 이미지가 있을 때,
      * $p(x) = \Pi_{i=1}^{n^2} p(x_{i,R}  \|  x_{< i}) p(x_{i,B}  \|  x_{< i}, x_{i, R}) p(x_{i,B}  \|  x_{< i}, X_{i,R}, X_{i, G})$
        * $p(x_{i,R}  \|  x_{< i})$ : i 번째 픽셀의 R 에 대한 확률​
        * $p(x_{i,B}  \|  x_{< i}, x_{i, R})$ : i 번째 픽셀의 G 에 대한 확률
        * $p(x_{i,B}  \|  x_{< i}, X_{i,R}, X_{i, G})$ : i 번째 픽셀의 B 에 대한 확률
    * 특징
      * auto-regressive model을 fully connected layer로 만든 것이 아니라 RNN을 이용한다.
      * ordering 방법에 따라
        * ![image](https://user-images.githubusercontent.com/35680202/129298636-069ae7a5-c3df-4d94-9457-aecd639745a5.png)
        * Row LSTM : 위쪽에 있는 정보 활용
        * Diagonal BiLSTM : 이전 정보들을 다 활용

## [10강] Generative Models 2

* **Variational Auto-encoder(VAE)**
  * Variational inference (VI) : posterior distribution 을 제일 잘 근사할 수 있는 variational distribution 을 찾는 일련의 과정
    * Posterior distribution $p_{\theta} (z \| x)$​
      * observation 이 주어졌을 때, 내가 관심있어하는 random variable의 확률분포 (이 반대를 likelihood 라고 부른다.) 
      * $z$​​는 latent vector(잠재벡터)
    * Variational distribution $q_{\phi} (z \| x)$​​ 
      * 일반적으로 posterior distribution을 계산하기 불가능할 때가 많다.
      * 학습&최적화를 통해 posterior distribution 를 근사하는 분포가 variational distribution
    * 방법
      * 마치 target을 모르는데 loss function을 찾고자 하는 것
      * KL divergence 를 최소화하는 variational distribution을 찾는다.
      * $\ln p_{\theta}(D)$
        * $= E_{q_{\phi}(z \| x)} [\ln p_{\theta}(x)]$​​
        * $= E_{q_{\phi}(z \| x)} [\ln \frac{p_{\theta}(x, z)}{p_{\theta}(z \| x)}]$​
        * $= E_{q_{\phi}(z \| x)} [\ln \frac{p_{\theta}(x, z) q_{\phi}(z \| x)}{q_{\phi}(z \| x) p_{\theta}(z \| x)}]$​​
        * $= E_{q_{\phi}(z \| x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \| x)}] + E_{q_{\phi}(z \| x)} [\ln \frac{q_{\phi}(z \| x)}{p_{\theta}(z \| x)}]$​​​​
        * $= E_{q_{\phi}(z \| x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \| x)}] + D_{KL}(q_{\phi}(z \| x)  \|  \|  p_{\theta}(z \| x))$
          * ELBO(↑) : $E_{q_{\phi}(z \| x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \| x)}]$​
          * Objective(↓) : $D_{KL}(q_{\phi}(z \| x)  \|  \|  p_{\theta}(z \| x))$​​​
      * VI는 ELBO를 최대화시킴으로써 (intractable한) Objective를 최소화시킨다.
    * ELBO(Evidence of Lower BOund)
      * $E_{q_{\phi}(z \| x)} [\ln \frac{p_{\theta}(x, z)}{q_{\phi}(z \| x)}]$​​​
        * $= \int \ln \frac{p_{\theta}(x \| z) p(z)}{q_{\phi}(z \| x)} q_{\phi}(z \| x) dz$
        * $= E_{q_{\phi}(z \| x)} [p_{\theta}(x \| z)] - D_{KL}(q_{\phi}(z \| x)  \|  \|  p(z))$
          * Reconstruction Term : $E_{q_{\phi}(z \| x)} [p_{\theta}(x \| z)]$ => 인코더를 통해서 x를 latent space로 보냈다가 다시 디코더로 돌아오는 auto-encoder의 reconstruction loss를 줄이는 것이 Reconstruction Term
          * Prior Fitting Term : $D_{KL}(q_{\phi}(z \| x)  \|  \|  p(z))$ => x 들을 latent space로 올려놓았을 때 점들이 이루는 분포가 내가 가정하는 사전분포(prior distribution)와 비슷하게 만들어주는 Term
    * 한계
      * intractable model 이다. (implicit model)
      * 미분 가능한 prior fitting term 을 사용해야 하므로, 다양한 latent prior distribution을 사용할 수 없다. (그래서 대부분의 경우 isotropic Gaussian 을 사용한다.)
* **Adversarial Auto-encoder(AAE)**
  * 방법
    * GAN을 사용해서 latent distribution 사이의 분포를 맞춰주는 것
    * Variational Autoencoder의 prior fitting term을 GAN의 objective로 바꿔버린 것
* **Generative Adversarial Network(GAN)**
  * ![image](https://user-images.githubusercontent.com/35680202/129310604-d6cc097f-5461-47ea-a226-6efba53e6a59.png)
  * $\underset{G}{\min} \underset{D}{\max} V(D, G) = E_{x \sim p_{data}(x)} [\log D(x)] + E_{z \sim p_z(z)} [\log(1 - D(G(z)))]$​
  * A two player minimax game between generator and discriminator
  * GAN Objective
  * For discriminator :
    * ![image](https://user-images.githubusercontent.com/35680202/129356148-dd5facb3-b7cc-4fd5-a755-0f5e146649fb.png)
  * For generator :
    * ![image](https://user-images.githubusercontent.com/35680202/129356264-918326bb-3f1e-4bfc-ad9a-ef9370ad641b.png)
  * implicit model
* **DCGAN**
  * GAN 이 MLP를 이용했다면 DCGAN 에서는 이미지 도메인으로 했다.
  * Deconvolution layer로 generator를 만들었다.
  * 여러 좋은 테크닉 사용 : leaky ReLU, 적절한 하이퍼파라미터 등
* **Info-GAN**
  * ![image](https://user-images.githubusercontent.com/35680202/129311554-f7c1f65c-7ec6-4244-8dbd-e5cdafdca39c.png)
  * class c 라는 auxiliary class를 랜덤하게 집어넣는다. (랜덤한 one-hot 벡터)
  * 마치 multi-modal distribution을 학습하는 것을 c라는 벡터를 통해서 잡아주는 역할
* **Text2Image**
  * 문장이 주어지면 이미지를 만드는 것
* **Puzzle-GAN**
  * 이미지 안의 subpatch 들이 있으면, 원래 이미지로 복원하는 것
* **CycleGAN**
  * GAN 구조를 사용하지만 이미지 사이의 도메인을 바꿀 수 있는 것
  * Cycle-consistency loss : 꼭 알아두기!
    * ![image](https://user-images.githubusercontent.com/35680202/129312171-87e97416-f4ee-4d6a-89f3-c257155ffe39.png)
    * GAN 구조가 2개가 들어감
* **Star-GAN**
  * 이미지를 단순히 다른 도메인으로 바꾸는 것이 아니라 내가 control 할 수 있게 하는 것
* **Progressive-GAN**
  * 고차원의 이미지를 잘 만들 수 있는 방법론
  * 4x4 부터 시작해서 1024x1024 까지 고해상도 이미지로 점점 늘려나가면서 학습하는 것
