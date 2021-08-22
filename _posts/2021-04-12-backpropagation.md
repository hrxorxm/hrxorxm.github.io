---
layout: post
title:  "[통계적 기계학습] 6. Backpropagation"
subtitle:   "역전파 계산과 구현 방법에 대해 알아보자 (Self-check 有)"
categories: "Statistical-Machine-Learning"
tags: [Backpropagation]
use_math: true
---

# Backpropagation

- 목차
    - [목표](#목표)
    - [Backprop with scalars](#backprop-with-scalars)
    - [Backprop with tensors](#backprop-with-tensors)
    - [Advanced topics](#advanced-topics)
    - [Self-check](#self-check)

## 목표
* $W^{(t+1)} = W^{(t)} - \eta_t \nabla L(W^{(t)})$
* (Bad) Idea : Derive $\nabla_W L$ on paper
  * lots of matrix calculus, need lots of paper
  * not modular (loss function 이 바뀔 때마다 다시 계산..)
  * not feasible for very complex models
* (Better) Idea : Computational Graphs
  * 내가 원하는 연산을 그래프로 그린다.
  * `Backpropagation` : 내가 정의한 연산의 반대 방향으로 gradient를 구한다.

## Backprop with scalars
* $f(x, y, z) = (x + y)z$ 에서 $\frac{\partial f}{\partial x}\mid_{(x,y,z)=(-2, 5, -4)}$ 을 구해보자
  * `Forward pass` : Compute outputs
    * $q = x + y$, $f = qz$
  * `Backward pass` : Compute derivatives
    * $\frac{\partial f}{\partial x}$, $\frac{\partial f}{\partial y}$, $\frac{\partial f}{\partial z}$ 를 구해야함
      1. $\frac{\partial f}{\partial f} = 1$
      2. $\frac{\partial f}{\partial z} = q = 3$
      3. $\frac{\partial f}{\partial q} = z = -4$
      4. $\frac{\partial f}{\partial y} = \frac{\partial q}{\partial y} \frac{\partial f}{\partial q} = 1 \cdot z = -4$
      5. $\frac{\partial f}{\partial x} = \frac{\partial q}{\partial x} \frac{\partial f}{\partial q} = 1 \cdot z = -4$
* 현재 노드 : $z = f(x, y)$
  * `Upstream Gradient`
    * 이미 계산된 값
    * $\frac{\partial L}{\partial z}$
  * `Local Gradients`
    * 새로 계산해야하는 값
    * $\frac{\partial z}{\partial x}$, $\frac{\partial z}{\partial y}$
  * `Downstream Gradients`
    * 구하고자 하는 값
    * $[Downstream] = [Local] * [Upstream]$ (`Chain Rule`)
    * $\frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \frac{\partial L}{\partial z}$
    * $\frac{\partial L}{\partial y} = \frac{\partial z}{\partial y} \frac{\partial L}{\partial z}$
* 다른 예제 : $f(x,w) = \frac{1}{1 + e^{-(w_0 x_0 + w_1 x_1 + w_2)}} = softmax(0, w_0 x_0 + w_1 x_1 + w_2)$ 의 첫번째 요소
  * Sigmoid : 
    * $\sigma (x) = \frac{1}{1 + e^{-x}}$
    * $z = sigmoid(x) := \sigma (x)$
  * Sigmoid local gradient : 
    * $\frac{\partial}{\partial x} [ \sigma (x) ] = \frac{e^{-x}}{(1+e^{-x})^2} = (1 - \sigma (x)) \sigma (x)$
    * $\frac{\partial z}{\partial x} = (1 - z)z$

### Implementation
* Patterns in Gradient Flow
  * **add** gate : gradient distributor
    * $z = x + y$
    * $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial z}$
    * $\frac{\partial f}{\partial y} = \frac{\partial f}{\partial z}$
  * **copy** gate : gradient adder
    * $\phi(w) = (w, w)$
    * $\frac{\partial}{\partial w} \phi(w) = (1, 1)$
    * $\frac{\partial f}{\partial w} = \frac{\partial(y, z)}{\partial w} \cdot (\frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}) = (1, 1) \cdot (4, 2) = 4 + 2 = 6$ (내적)
  * **mul** gate : swap multiplier
    * $z = xy$
    * $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial z} * y$
    * $\frac{\partial f}{\partial y} = x * \frac{\partial f}{\partial z}$
  * **max** gate : gradient router
    * $z = max(x, y)$
    * $\frac{\partial z}{\partial x}= 1$ if $x > y$ else $0$
    * $\frac{\partial z}{\partial y}= 1$ if $y > x$ else $0$
    * if $y > x$
      * $\frac{\partial f}{\partial x} = 0$
      * $\frac{\partial f}{\partial y} = \frac{\partial f}{\partial z}$
    * 더 큰 방향으로 gradient 를 흘려보내주는 gradient router 같은 역할
* Backprop Implementation : Modular API
  * 하나하나의 computational graph 에 대해서 class 로 정의하고, forward() 와 backward() 하는 모듈을 만들어놓음
  * Example : PyTorch Autograd Functions

## Backprop with tensors
### Backprop with vectors
* Vector Derivatives
  * $x \in R$, $y \in R$
    * Regular derivative
    * $\frac{\partial y}{\partial x} \in R$
  * $x \in R^N$, $y \in R$
    * Derivative is **Gradient**
    * $\frac{\partial y}{\partial x} \in R^N$
    * ${(\frac{\partial y}{\partial x})}_{n} = \frac{\partial y}{\partial x_n}$
  * $x \in R^N$, $y \in R^M$
    * Derivative is **Jacobian**
    * $\frac{\partial y}{\partial x} \in R^{N \times M}$
    * ${(\frac{\partial y}{\partial x})}_{n, m} = \frac{\partial y_m}{\partial x_n}$

* 설명
  * 가정
    * 현재 노드 : $z = f(x, y)$
    * $x \in R^{D_x}$, $y \in R^{D_y}$, $z \in R^{D_z}$
    * $L$ : scalar
  * Upstream Gradient
    * $\frac{\partial L}{\partial z} \in R^{D_z}$
  * Local Jacobian matrices
    * $\frac{\partial z}{\partial x} \in R^{D_x \times D_z}$
    * $\frac{\partial z}{\partial y} \in R^{D_y \times D_z}$
  * Downstream Gradients (Matrix-vector multiply)
    * $\frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \frac{\partial L}{\partial z} \in R^{D_x}$ 
    * $\frac{\partial L}{\partial y} = \frac{\partial z}{\partial y} \frac{\partial L}{\partial z} \in R^{D_y}$
  * **Jacobian is sparse**
    * Never **explicitly** form Jacobian
    * instead use **implicit** multiplication
      * ex) ReLU : ${(\frac{\partial L}{\partial x})}_{i} = {(\frac{\partial L}{\partial y})}_{i}$ if $x_i > 0$ otherwise 0

### Backprop with Matrices (or Tensors)
* 설명
  * 가정
    * 현재 노드 : $z = f(x, y)$
    * $x \in R^{D_x \times M_x}$, $y \in R^{D_y \times M_y}$, $z \in R^{D_z \times M_z}$
      * reshape(-1) 해서 벡터로 만든걸 사용한다.
    * $L$ : scalar
  * Upstream Gradient
    * $\frac{\partial L}{\partial z} \in R^{D_z \times M_z}$
  * Local Jacobian matrices
    * $\frac{\partial z}{\partial x} \in R^{(D_x \times M_x) \times (D_z \times M_z)}$
    * $\frac{\partial z}{\partial y} \in R^{(D_y \times M_y) \times (D_z \times M_z)}$
  * Downstream Gradients (Matrix-vector multiply)
    * $\frac{\partial L}{\partial x} = \frac{\partial z}{\partial x} \frac{\partial L}{\partial z} \in R^{D_x \times M_x}$ 
    * $\frac{\partial L}{\partial y} = \frac{\partial z}{\partial y} \frac{\partial L}{\partial z} \in R^{D_y \times M_y}$

* 예제 : Matrix Multiplication
  * Matrix Multiply $y = xw$
    * $x \in R^{N \times D}$, $w \in R^{D \times M}$, $y \in R^{N \times M}$
    * $y_{n,m} = \sum_{d} x_{n,d} w_{d,m}$
  * Jacobians
    * $\frac{\partial y}{\partial x} \in R^{(N \times D) \times (N \times M)}$
    * $\frac{\partial y}{\partial w} \in R^{(D \times M) \times (N \times M)}$
    * Each Jacobian takes lots of memory, so Must work with them implicitly!
  * 방법 유도
    * $\frac{\partial L}{\partial x_{1,1}} = (\frac{\partial y}{\partial x_{1,1}}) \cdot (\frac{\partial L}{\partial y}) = (w_{1,:}) \cdot (\frac{\partial L}{\partial y_{1,:}})$ (내적)
      * $y_{1,1} = x_{1,1} w_{1,1} + x_{1,2} w_{2,1} + x_{1,3} w_{3,1}$
        * $\frac{\partial y_{1,1}}{\partial x_{1,1}} = w_{1,1}$
      * $y_{1,2} = x_{1,1} w_{1,2} + x_{1,2} w_{2,2} + x_{1,3} w_{3,2}$
        * $\frac{\partial y_{1,2}}{\partial x_{1,1}} = w_{1,2}$
      * $y_{2,1} = x_{2,1} w_{1,1} + x_{2,2} w_{2,1} + x_{2,3} w_{3,1}$
        * $\frac{\partial y_{2,1}}{\partial x_{1,1}} = 0$
    * $\frac{\partial L}{\partial x_{2,3}} = (\frac{\partial y}{\partial x_{2,3}}) \cdot (\frac{\partial L}{\partial y}) = (w_{3,:}) \cdot (\frac{\partial L}{\partial y_{2,:}})$
    * 일반적으로, $\frac{\partial L}{\partial x_{i,j}} = (\frac{\partial y}{\partial x_{i,j}}) \cdot (\frac{\partial L}{\partial y}) = (w_{j,:}) \cdot (\frac{\partial L}{\partial y_{i,:}})$
  * 결론
    * $\frac{\partial L}{\partial x} = (\frac{\partial L}{\partial y}) w^T$ ($[N \times D] = [N \times M] [M \times D]$)
    * $\frac{\partial L}{\partial w} = x^T (\frac{\partial L}{\partial y})$ ($[D \times M] = [D \times N] [N \times M]$)
    * Jacobian matrice 를 explicit 하게 쓰지 않고, implicit 하게 작은 matrix 들간의 연산으로 쓸 수 있음

## Advanced topics
### Backpropagation : Another View
* `Reverse-Mode Automatic Differentiation` (후진 자동 미분)
  * $x_0 (\in R^{D_0}) \rightarrow x_1 (\in R^{D_1}) \rightarrow x_2 (\in R^{D_2}) \rightarrow x_3 (\in R^{D_3}) \rightarrow L \in R$ (scalar)
  * Chain rule
    * $\frac{\partial L}{\partial x_0} = (\frac{\partial x_1}{\partial x_0}) (\frac{\partial x_2}{\partial x_1}) (\frac{\partial x_3}{\partial x_2}) (\frac{\partial L}{\partial x_3})$
    * $[D_0 \times D_1] [D_1 \times D_2] [D_2 \times D_3] D_3$
  * Matrix multiplication is **associative** : 결합법칙 성립
  * matrix-matrix products 를 피하기 위해 right-to-left 순서로 계산하기 (matrix-vector 연산만 필요)
* `Forward-Mode Automatic Differentiaction` (전진 자동 미분)
  * $a (\in R) \rightarrow x_0 (\in R^{D_0}) \rightarrow x_1 (\in R^{D_1}) \rightarrow x_2 (\in R^{D_2}) \rightarrow x_3 (\in R^{D_3})$ 
    * 애니메이션에서 눈, 유체 등을 모델링하기 위한 작업이라고 생각해보자
    * $a$ : 눈의 점성, 탄성
    * $x_3$ : 픽셀
  * Chain rule
    * $\frac{\partial x_3}{\partial a} = (\frac{\partial x_0}{\partial a}) (\frac{\partial x_1}{\partial x_0}) (\frac{\partial x_2}{\partial x_1}) (\frac{\partial x_3}{\partial x_2})$
    * $D_0 [D_0 \times D_1] [D_1 \times D_2] [D_2 \times D_3]$
  * 이번에는 left-to-right 로 계산하는 것이 효율적이다.
  * but PyTorch 나 TensorFlow 에 구현되어있지 않다. (matlab에 있을수도)
### Backprop : Higher-Order Derivatives
* Example : Regularization to penalize the norm of the gradient
* second derivative 까지 computational graph 를 그린 후에 Backprop 을 하면 된다.
* PyTorch 나 TensorFlow 에 구현되어 있다.


# Self-check

## 문제1
* explicit bias term이 있는 상태에서의 선형변환, 즉 z_i = w^T x_i + b를 N개의 sample (I = 0, … , N-1)에 대하여 동시에 계산하려 합니다 (for 구문을 이용하지 않고 벡터/행렬 연산을 이용하여). 아래 computational graph에서 파란 상자의 수식을 유도하십시오. 

![image](https://user-images.githubusercontent.com/35680202/114642883-9aac8580-9d0f-11eb-908d-164b8c54c2cc.png)

* 답 : copy gate 여서 gradient adder 역할을 한다.
  * $X \in R^{N \times D}$, $W \in R^{D \times H}$, $b \in R^{H}$
  * $Q = XW$ $\rightarrow$ $Z = Q + B$ $\rightarrow$ $\in R^{N \times H}$
    * $B = [b^T, \dotsc, b^T] \in R^{N \times H}$
  * Upstream gradient
    * $\frac{\partial \Phi}{\partial Z} \in R^{N \times H}$
  * Local gradient
    * $\frac{\partial Z}{\partial Q} = I$
    * $\frac{\partial Z}{\partial B} = I$
  * Downstream gradient
    * $\frac{\partial \Phi}{\partial B} = \frac{\partial \Phi}{\partial Z}$
    * $B$ 가 $b$ 를 copy 한 행렬이기 때문에, copy gate 에 의해 gradient adder 역할을 한다.
      * ${(\frac{\partial B}{\partial b})}_{i,j} = 1$, ${(\frac{\partial B}{\partial b})}_{j} = [1, \dotsc, 1] \in R^{N}$
      * ${(\frac{\partial \Phi}{\partial b})}_{j} = {(\frac{\partial B}{\partial b})}_{j} \cdot {(\frac{\partial \Phi}{\partial Z})}_{j}$

## 문제2
* softmax loss function (without normalizing by sample size)를 N개의 sample (I = 0, … , N-1)에 대하여 동시에 계산하려 합니다 (for 구문을 이용하지 않고 벡터/행렬 연산을 이용하여). 아래 computational graph에서 파란 상자의 수식을 유도하십시오.

![image](https://user-images.githubusercontent.com/35680202/114642981-caf42400-9d0f-11eb-81de-a67b04238da8.png)

* 답 : 
  * $L_i = - log(\frac{exp(S_{i y_i})}{\sum_{k=0}^{C-1} exp(S_{ik})}) = - S_{i y_i} + log(\sum_{k=0}^{C-1} exp(S_{ik}))$
  * Upstream gradient
    * $\frac{\partial \Phi}{\partial L}$
  * Local gradient
    * $\frac{\partial L}{\partial L_i} = 1$
    * $\frac{\partial L_i}{\partial S_{ij}}$
      * $= \frac{exp(S_{ij})}{\sum_{k=0}^{C-1} exp(S_{ik})}$ if $j \neq y_i$
      * $= (\frac{exp(S_{ij})}{\sum_{k=0}^{C-1} exp(S_{ik})} - 1)$ if $j = y_i$
  * Downstream gradient
    * $\frac{\partial \Phi}{\partial S_{ij}} = \frac{\partial L_i}{\partial S_{ij}} \cdot \frac{\partial L}{\partial L_i} \cdot \frac{\partial \Phi}{\partial L}$
      * $= \frac{exp(S_{ij})}{\sum_{k=0}^{C-1} exp(S_{ik})} \cdot \frac{\partial \Phi}{\partial L}$ if $j \neq y_i$
      * $= (\frac{exp(S_{ij})}{\sum_{k=0}^{C-1} exp(S_{ik})} - 1) \cdot \frac{\partial \Phi}{\partial L}$ if $j = y_i$

## 문제3
* multiclass svm loss (without normalizing by sample size)를 N개의 sample (I = 0, … , N-1)에 대하여 동시에 계산하려 합니다 (for 구문을 이용하지 않고 벡터/행렬 연산을 이용하여). 아래 computational graph에서 파란 상자의 수식을 유도하십시오.

![image](https://user-images.githubusercontent.com/35680202/114643062-ebbc7980-9d0f-11eb-929c-c18962c2d933.png)

* 답 : 
  * $L_i = \sum_{i=0}^{N-1} \sum_{k \neq y_i} max(0, S_{ik} - S_{i y_i} + 1)$
  * Upstream gradient
    * $\frac{\partial \Phi}{\partial L}$
  * Local gradient
    * $\frac{\partial L}{\partial L_i} = 1$
    * $\frac{\partial L_i}{\partial S_{ij}}$
      * $= I(S_{ij} - S_{i y_i} + 1 > 0)$ if $j \neq y_i$
      * $= - \sum_{k \neq j} I(S_{ik} - S_{i y_i} + 1 > 0)$ if $j = y_i$
  * Downstream gradient
    * $\frac{\partial \Phi}{\partial S_{ij}} = \frac{\partial L_i}{\partial S_{ij}} \cdot \frac{\partial L}{\partial L_i} \cdot \frac{\partial \Phi}{\partial L}$
      * $= I(S_{ij} - S_{i y_i} + 1 > 0) \cdot \frac{\partial \Phi}{\partial L}$ if $j \neq y_i$
      * $= - \sum_{k \neq j} I(S_{ik} - S_{i y_i} + 1 > 0) \cdot \frac{\partial \Phi}{\partial L}$ if $j = y_i$