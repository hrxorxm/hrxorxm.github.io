---
layout: post
title:  "[통계적 기계학습] 4. Optimization"
subtitle:   "최적화 알고리즘의 종류를 알아보자 (Self-check 有)"
categories: "Statistical-Machine-Learning"
tags: [Optimization]
use_math: true
---

# Optimization

- 목차
    - [목표](#목표)
    - [Why Gradient is required instead of Random search?](#why-gradient-is-required-instead-of-random-search)
    - [Gradient Descent Algorithm](#gradient-descent-algorithm)
    - [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
    - [Variants of SGD](#variants-of-sgd)
    - [Why not second-order optimization?](#why-not-second-order-optimization)
    - [Self-check](#self-check)

## 목표 
* `Optimization` : 최적화, 함수의 최댓값이나 최솟값을 찾는 문제
  * (Numerical) Optimization : 수치적 최적화, 컴퓨터의 도움을 받아서 값을 찾는 것
* 최적의 $W$ 로 어떻게 찾아갈까?
  * $w^* = argmin_{w} L(w)$
    * $L(w)$ 의 최소화원 찾기
    * 일반적인 함수 최소화 문제 : 미분 여러번 하고 그래프 그려서 찾을 수 있음
  * 범용적인 알고리즘
    * Gradient descent (`full-batch`)
      * $f(x)$ 의 그레디언트만 이용해서 최솟값을 찾아나가는 것
      * first-order 알고리즘 (미분을 한번만 함) 중에 하나
    * second-order optimization
  * 딥러닝에 특화된 알고리즘
    * Gradient descent (`stochastic`)
      * Momentum
      * Nesterov momentum
      * AdaGrad
      * RMSProp
      * Adam

## Why Gradient is required instead of Random search?
1. Random Search (bad idea)
   * ex) $[0, 1]$ 에서 $0.01$ 간격으로 grid 를 채우기 위해서 필요한 점의 개수는?
     * $w \in R^1$ 일 때 : $100$ 개
     * $w \in R^{3 \times 32 \times 32 + 1}$ 일 때 : $100^{3073}$ 개
   * dimension이 커질 때 일일이 서치해야하는 점들의 수가 너무 많아진다.
   * 이런 것들을 무시하고 일정 개수만 뽑아서 비교한다고 하면, 충분히 다양하게 탐색해볼 수가 없다.
   * SOTA (State Of The Art) : 현재 최고 수준의 결과
2. `Follow the slope : gradient`
   * gradient : 경사, 방향 미분
     * $x \in R^1$ 일 때 : $\nabla{f(x)} = \frac{df(X)}{dx} = {\lim}_{h \to 0} \frac{f(x + h) - f(x)}{h}$
     * $\nabla{f(x)}^T d$ : $x$ 라는 점에서 $d$ 방향으로의 순간 변화율
       * 코시-슈바르츠 부등식에 의해, $d = \nabla{f(x)}$일 때 가장 높은 순간 변화율을 볼 수 있다. 
       * 따라서 내려가는 방향으로 가려면 $- \nabla{f(x)}$ 방향으로 가면 된다.

## Gradient Descent Algorithm
* $w^{(t+1)} = w^{(t)} - \eta_{t} \nabla{L(w^{(t)})}$
  * $\eta_{t}$ : step size, 미리 정해둔 scalar
  * gradient 의 negative 한 방향으로 간다.
* Calculating gradients
  * `analytical`
    * 만약, $f(x) = x^3$ 일 때, $f\prime(x) = 3x^2$ 를 구해서 $f\prime(4)$ 를 구하는 방식
    * 장점 : exact, fast = 정확하고 빠르다.
    * 단점 : error-prone = 에러에 취약하다.
  * `numerical`
    * $f\prime(4) = {\lim}_{h \to 0} \frac{f(4 + h) - f(4)}{h}$ 에서 $h = 0.00001$ 과 같은 걸 대입해서 구함
    * $w \in R^H$ 일 때, $O(H)$ 의 속도를 가짐(느림)
    * 장점 : easy to write = 코드를 짜기 쉽다.
    * 단점 : approximate, slow = 근사값이고 느리다.
  * gradient check : 딥러닝 구현시에는 analytic gradient 를 사용하지만, 점검을 위해서는 numerical gradient 를 사용한다.

## Stochastic Gradient Descent (SGD)
* 용어
  * Vanilla gradient descent
    * 아무런 튜닝하지 않은 알고리즘
    * Hyperparameters (training set이 아닌 validation set으로 결정)
      * Weight initialization method
      * Number of steps
      * Learning rate
  * `Stochastic Gradient Descent (SGD)`
    * 직관 : 대수의 법칙 (Law of large numbers)
      * $E(X) \approx \frac{1}{100} \sum_{i=1}^{100} X_i$
    * Hyperparameters
      * Weight initialization method
      * Number of steps
      * Learning rate
      * **Batch size**
      * **Data sampling**
* `Stochastic Gradient Descent algorithm`
  * $w^{(t+1)} = w^{(t)} - \eta_{t} \nabla{L(w^{(t)})}$ ... (1)
  * when loss function is of form $L(w) = \frac{1}{N} \sum_{i=1}^{N} L_i(w) + R(w)$
  * Run (1), but calculate $\nabla L(w^{(t)})$ by its approximation ($\frac{1}{M} \sum_{i=1}^{M} L_i(w) + R(w)$, i 는 randomly picked M개)
    * batch size : M
    * minibatch : randomly 골라진 M개의 $\{(x_i, y_i)\}$
* Problems with (S)GD
  * Jittering : $\eta_t$ 의 영향
    * 장축과 단축에 극단적이 차이가 있는 경우에 심해진다.
    * Hessian Matrix : 어떤 함수의 이계도함수를 행렬로 표현한 것
    * condition number : minimum eigenvalue 와 maximum eigenvalue 의 비율, 이 값이 클수록 학습이 수렴하는데에 오래걸림
  * Local minimum or saddle point(안장점) : Global minimum 을 찾는 것이 보장된 알고리즘은 없음
  * Noisy (SGD 문제점)

## Variants of SGD
* Momentum-based : descent direction 에 지금까지의 누적된 gradient 를 반영하는 것 (속도를 조정한다)
  * `Momentum` (관성)
    * [SGD + Momentum](https://hyunw.kim/blog/2017/11/01/Optimization.html)
      * $- \alpha$ 를 먼저 곱해서 넣는 경우
        * $v_{t+1} = \rho v_t - \alpha \nabla f(x_t)$
        * $x_{t+1} = x_t + v_{t+1}$
      * $- \alpha$ 를 나중에 곱해서 넣는 경우 (아래 있는 Adam 은 이걸 참조)
        * $v_{t+1} = \rho v_t + \nabla f(x_t)$
        * $x_{t+1} = x_t - \alpha v_{t+1}$
      * $\rho$ : 모멘텀 계수 (learning rate $\alpha$ 와 같이 상수이다.)
        * 곡면이 얼마나 매끄러운지를 나타냄
        * 매끄러울수록 값이 큼 ($\rho < 1$)
        * 매끄럽다 - 마찰이 적다 - 수치가 높을수록 과거 영향이 크다.
        * 보통 0.9 나 0.99 로 설정한다.
    * 직관 : 방향이 누적되어서 들어가게 됨 (velocity)
    * 장점
      * Local minimum에 빠졌을 때 빠져나갈 수 있다.
      * Noise 를 평균내는 효과가 있다. (방향이 스무스해진다.)
    * 단점
      * 일반적으로 훨씬 더 jittering 할 수 있다고 알려짐
  * `Nesterov momentum`
    * $v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)$
    * $x_{t+1} = x_t + v_{t+1}$
    * 직관 : 실제로 가게 될 지점 근처에서 gradient를 계산해야 현상을 더 잘 볼 수 있다.
* learning rate 를 조정하는 것
  * `AdaGrad`
    * $r_{t+1} = r_t + \nabla L(w) * \nabla L(w)$ ($r_{t+1} \in R^H$)
    * $w_{t+1} = w_t - \frac{\eta_t}{\sqrt{r_{t+1}}} \nabla L(w)$
    * 축에 따라 곱해지는 learning rate 가 달라짐
    * 내가 덜 갔던 방향으로는 gradient가 덜 쌓였기 때문에 그 방향으로는 많이 가게 된다.
    * Adaptive : learning rate 를 상수로 고정하는 것이 아니라, learning rate가 각 축마다 (이전에 안 배운 방향은 세게 / 이전에 많이 간 방향은 작게) 적용된다. 즉, learning rate 가 data adaptive 하게 결정된다.
  * `RMSProp` : Adaptive Gradient 에서 발전, "Leak AdaGrad"
    * $r_{t+1} = \beta r_t + (1 - \beta) \nabla f(x_t) * \nabla f(x_t)$ ($0 < \beta < 1$)
    * 직관 : 
      * AdaGrad : $r_t \rightarrow \infty$ as $t \rightarrow \infty$
      * RMSProp : $\mid r_t \mid \leq c$ as $t \rightarrow \infty$ (단, c 는 상수)
      * 차이점 : AdaGrad 에서는 $t$ 가 커질수록 $r_{t+1}$ 이 커지기만 하는데, RMSProp 에서는 일정한 스케일로 유지가 된다.
* descent direction 과 learning rate 를 둘 다 조정하는 것
  * `Adam` (almost) : RMSProp + Momentum
    * 우선, $0 < \beta_1 <1$, $0 < \beta_2 < 1$
    * **descent direction** (Momentum, 누적된 gradient값을 저장해놓는 속도 파트)
      * moment1 (first moment)
      * $v_{t+1} = \beta_1 v_t + (1 - \beta_1) \nabla L(w_t)$
    * Adaptive **learning rate** (AdaGrad, RMSProp)
      * moment2 (second mement)
      * $r_{t+1} = \beta_2 r_t + (1 - \beta_2)(\nabla L(w_t) * \nabla L(w_t))$
    * (Bias correction 하면)
      * $v_{t+1} = v_{t+1} / (1 - \beta_1^{t})$
      * $r_{t+1} = r_{t+1} / (1 - \beta_2^{t})$
    * 따라서, $w_{t+1} = w_t - \frac{\eta_t}{\sqrt{r_{t+1}}} v_{t+1}$
    * (참고) Bias correction
      * Motivation : 처음에 0이 되는 term이 있고, $\beta$를 곱해서 1보다 작은 값을 곱해버리니까 충분히 많이 가지 못한다.
      * unbias = mement / $(1 - \beta^{t})$
      * 손해를 볼 수 있는 크기만큼 (t번 쌓인 만큼) 다시 분모로 나눠줘서 스케일을 원래대로 살리는 테크닉
    * 보통, $\beta_1 = 0.9$, $\beta_2 = 0.999$, learning_rate = 1e-3, 5e-4, 1e-4
    * Optimization 알고리즘은 보통은 그냥 Adam 쓴다.

## Why not second-order optimization?
* First-Order Optimization : gradient(linear approximation) 계산해서 얼만큼 갈건지 결정
  * Taylor 1차 근사 : $f(x) \approx f(x_0) + \nabla f(x_o)^T (x - x_0)$
* Second-Order Optimization : gradient 뿐만 아니라 Hessian 을 사용하여 2차 함수를 만든 후 최솟값을 찾아낼 것
  * Taylor 2차 근사 : $f(x) \approx f(x_0) + \nabla f(x_0)^T (x - x_0) + \frac{1}{2} (x - x_0)^T \nabla^2 f(x_0) (x - x_0)$
    * 2차식이면 최솟값을 구할 수 있다.
      * $\frac{1}{2}ax^2 + bx +c$ 이면, $x = - \frac{b}{a}$ 일 때 최소
    * $x_{t+1} = x_t - \nabla^2 f^{-1}(x_t) \nabla f(x_t)$
      * Negative direction 에다가 Hessian Matrix 의 Inverse 를 통째로 곱해준다.
  * Second-Order Taylor Expansion:
    * $L(w) \approx L(w_0) + (w - w_0)^T \nabla_w L(w_0) + \frac{1}{2}(w - w_0)^T H_w L(w_0)(w - w_0)$
    * $w^* = w_0 - H_w L(w_0)^{-1} \nabla_w L(w_0)$
  * 장점 : 
    * Jittering 문제에 강하다. 
    * learning rate가 필요없다. 함수곡선에 따라서 얼마나 갈지 자기가 알아서 결정하기 때문이다.
  * 단점 : Matrix Inverse 를 구하기 어렵다.
    * $H_w L(w_0) \in R^{H \times H}$ 일 때, Matrix Inverse 를 구하기 위한 시간 복잡도가 $O(H^3)$ 이다.
    * CIFAR10 을 예로 들어도, $H = 3073$ 이나 된다.
    * 실제로 H = (Tens or Hundreds of) Millions 정도 된다.
    * 따라서 Hessian Matrix의 근사를 계산하는 알고리즘들이 제안됨.
      * ex) BFGS, L-BFGS


# Self-check
## 문제1
* 실수값을 취하는 다변수함수 f(x)의 그라디언트를 컴퓨터로 계산할 수 있는 두 가지 방법을 논하시오. - 0.5 pages
* 답
  * `analytical`
    * 만약, $f(x) = x^3$ 일 때, $f\prime(x) = 3x^2$ 를 구해서 $f\prime(4)$ 를 구하는 방식
    * 장점 : exact, fast = 정확하고 빠르다.
    * 단점 : error-prone = 에러에 취약하다.
  * `numerical`
    * $f\prime(4) = {\lim}_{h \to 0} \frac{f(4 + h) - f(4)}{h}$ 에서 $h = 0.00001$ 과 같은 걸 대입해서 구함
    * $w \in R^H$ 일 때, $O(H)$ 의 속도를 가짐(느림)
    * 장점 : easy to write = 코드를 짜기 쉽다.
    * 단점 : approximate, slow = 근사값이고 느리다.
  * gradient check : 딥러닝 구현시에는 analytic gradient 를 사용하지만, 점검을 위해서는 numerical gradient 를 사용한다.

## 문제2
* Gradient descent (GD) algorithm은 무엇입니까? (사용하는 목적과 알고리즘 수식을 소개하세요). Stochastic gradient descent (SGD) algorithm은 무엇입니까 (GD와의 차이를 위주로 서술하세요)? 왜 훈련 데이터셋의 사이즈가 크면 GD보다 SGD를 더 고려합니까? - 0.5 pages
* 답
  * Gradient descent (GD) algorithm은 무엇입니까? (사용하는 목적과 알고리즘 수식을 소개하세요)
    * Random Search 는 dimension이 커질 때 일일이 서치해야하는 점들의 수가 너무 많아진다. 이런 것들을 무시하고 일정 개수만 뽑아서 비교한다고 하면, 충분히 다양하게 탐색해볼 수가 없다. 따라서 변수들을 적절히 업데이트하면서 학습하는 방법이 합리적이다.
    * $\nabla{f(x)}^T d$ : $x$ 라는 점에서 $d$ 방향으로의 순간 변화율
      * 코시-슈바르츠 부등식에 의해, $d = \nabla{f(x)}$일 때 가장 높은 순간 변화율을 볼 수 있다. 
      * 따라서 내려가는 방향으로 가려면 $- \nabla{f(x)}$ 방향으로 가면 된다.
    * $w^{(t+1)} = w^{(t)} - \eta_{t} \nabla{L(w^{(t)})}$
      * $\eta_{t}$ : step size, 미리 정해둔 scalar
      * gradient 의 negative 한 방향으로 간다.
  * Stochastic gradient descent (SGD) algorithm은 무엇입니까? (GD와의 차이를 위주로 서술하세요)
    * GD : $L(w) = \frac{1}{N} \sum_{i=1}^{N} L_i(w) + R(w)$
    * SGD : $L(w) = \frac{1}{M} \sum_{i=1}^{M} L_i(w) + R(w)$, i 는 randomly picked M개)
    * 따라서, Batch size 와 Data sampling 방법에 대한 hyperparameter 가 추가적으로 요구된다.
  * 왜 훈련 데이터셋의 사이즈가 크면 GD보다 SGD를 더 고려합니까?
    * GD 에서는 한번 step을 내딛을 때 전체 데이터에 대해 Loss Function을 계산해야 하므로 너무 많은 계산량이 필요하다. 특히 훈련 데이터셋의 사이즈가 클수록 메모리 안에서 계산되는 것이 부담스러워질 수 있다. 따라서 SGD 를 이용해서 batch size 만큼씩 loss function을 계산하여 업데이트 해나가는 것이 합리적이다.

## 문제3
* SGD의 변형 알고리즘을 다섯 개를 소개하세요. (알고리즘들이 등장한 motivation과 수식을 소개하세요. Nestrov momentum은 motivation을 적지 않으셔도 괜찮습니다.) - 1 page 
* 답
  * Problems with (S)GD
    * Jittering : $\eta_t$ 의 영향
    * Local minimum or saddle point(안장점) : Global minimum 을 찾는 것이 보장된 알고리즘은 없음
    * Noisy (SGD 문제점)
  * Momentum-based : descent direction 에 지금까지의 누적된 gradient 를 반영하는 것 (속도를 조정한다)
    * `Momentum` (관성)
      * SGD + Momentum
        * $- \alpha$ 를 먼저 곱해서 넣는 경우
          * $v_{t+1} = \rho v_t - \alpha \nabla f(x_t)$
          * $x_{t+1} = x_t + v_{t+1}$
        * $- \alpha$ 를 나중에 곱해서 넣는 경우 (아래 있는 Adam 은 이걸 참조)
          * $v_{t+1} = \rho v_t + \nabla f(x_t)$
          * $x_{t+1} = x_t - \alpha v_{t+1}$
        * $\rho$ : 모멘텀 계수 (learning rate $\alpha$ 와 같이 상수이다.), 수치가 높을수록 과거 영향이 크다.
      * 직관 : 방향이 누적되어서 들어가게 됨 (velocity)
      * 장점 : Local minimum에 빠졌을 때 빠져나갈 수 있고, Noise 를 평균내는 효과가 있다. (방향이 스무스해진다.)
      * 단점 : 일반적으로 훨씬 더 jittering 할 수 있다고 알려짐
    * `Nesterov momentum`
      * $v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)$
      * $x_{t+1} = x_t + v_{t+1}$
      * 직관 : 실제로 가게 될 지점 근처에서 gradient를 계산해야 현상을 더 잘 볼 수 있다.
  * learning rate 를 조정하는 것
    * `AdaGrad`
      * $r_{t+1} = r_t + \nabla L(w) * \nabla L(w)$ ($r_{t+1} \in R^H$)
      * $w_{t+1} = w_t - \frac{\eta_t}{\sqrt{r_{t+1}}} \nabla L(w)$
      * Adaptive : learning rate 를 상수로 고정하는 것이 아니라, learning rate가 각 축마다 (이전에 안 배운 방향은 세게 / 이전에 많이 간 방향은 작게) 적용된다. 즉, learning rate 가 data adaptive 하게 결정된다.
    * `RMSProp` : Adaptive Gradient 에서 발전, "Leak AdaGrad"
      * $r_{t+1} = \beta r_t + (1 - \beta) \nabla f(x_t) * \nabla f(x_t)$ ($0 < \beta < 1$)
      * 차이점 : AdaGrad 에서는 $t$ 가 커질수록 $r_{t+1}$ 이 커지기만 하는데, RMSProp 에서는 일정한 스케일로 유지가 된다.
  * descent direction 과 learning rate 를 둘 다 조정하는 것
    * `Adam` (almost) : RMSProp + Momentum
      * 우선, $0 < \beta_1 <1$, $0 < \beta_2 < 1$
      * **descent direction** (Momentum, 누적된 gradient값을 저장해놓는 속도 파트)
        * moment1 (first moment) : $v_{t+1} = \beta_1 v_t + (1 - \beta_1) \nabla L(w_t)$
      * Adaptive **learning rate** (AdaGrad, RMSProp)
        * moment2 (second mement) : $r_{t+1} = \beta_2 r_t + (1 - \beta_2)(\nabla L(w_t) * \nabla L(w_t))$
      * Bias correction : 처음에 0이 되는 term이 있고, \betaβ를 곱해서 1보다 작은 값을 곱해버리니까 충분히 많이 가지 못한다. 손해를 볼 수 있는 크기만큼 (t번 쌓인 만큼) 다시 분모로 나눠줘서 스케일을 원래대로 살리는 테크닉
        * $v_{t+1} = v_{t+1} / (1 - \beta_1^{t})$
        * $r_{t+1} = r_{t+1} / (1 - \beta_2^{t})$
      * 따라서, $w_{t+1} = w_t - \frac{\eta_t}{\sqrt{r_{t+1}}} v_{t+1}$

## 문제4
* f(x)의 최솟값을 구하기 위한 first-order method와 second-order method의 차이는 무엇입니까? 딥러닝에서 보통 first-order method만 사용하는 이유는 무엇입니까? - 0.5 pages
  * first-order method와 second-order method의 차이는 무엇입니까?
    * First-Order Optimization : gradient(linear approximation) 계산해서 얼만큼 갈건지 결정
    * Second-Order Optimization : gradient 뿐만 아니라 Hessian 을 사용하여 2차 함수를 만든 후 최솟값을 찾아내는 것
      * 장점 : Jittering 문제에 강하다. learning rate가 필요없다. 함수곡선에 따라서 얼마나 갈지 자기가 알아서 결정하기 때문이다.
  * 딥러닝에서 보통 first-order method만 사용하는 이유는 무엇입니까?
    * second- 의 단점 : 현실 데이터에서 Matrix Inverse 를 구하기 어렵다. Matrix Inverse 를 구하기 위한 시간 복잡도가 너무 커진다. (3승?)
    * 따라서 조금 느리더라도 계산복잡도가 적당한 first- 를 주로 사용한다.