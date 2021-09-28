---
layout: post
title:  "[파이토치 튜토리얼/허브 번역] 2. 허브 사이트 빌드 및 번역 연습"
subtitle:   "오픈소스 컨트리뷰션 아카데미 2주차"
categories: "OpenSource-Contribution"
tags: []
use_math: true
---

# Week2

## 1. 허브 사이트 빌드하기 (8/15)

### PyTorchKR 다운받기

```bash
$ git clone https://github.com/9bow/PyTorchKR
$ cd PyTorchKR
$ git submodule sync
$ git submodule update --init --recursive
```

### Windows10에서 우분투 사용하기

- [Windows 참가자를 위한 단순화된 설치](https://docs.microsoft.com/ko-kr/windows/wsl/install-win10#simplified-installation-for-windows-insiders)
  - 관리자권한으로 cmd열어서 `wsl --install`
- [Window10 WSL 로 Ubuntu 설치하기](https://videocube.tistory.com/entry/Window10에서-Ubuntu-설치하기)
  - Ubuntu 설치 후 다시시작
  - username 과 password 입력하면 끝

### 허브 사이트 빌드하기

- 패키지 설치하기
  - nvm : https://www.vultr.com/docs/install-nvm-and-node-js-on-ubuntu-20-04
  - rbenv, ruby-build : https://linoxide.com/how-to-install-ruby-on-rails-on-ubuntu-20-04/

```bash
$ nvm use 9.8
$ node -v
v9.8.0
$ npm install
$ rbenv versions
  2.5.8
* 2.5.9 (set by /home/user/workspaces/PyTorchKR/.ruby-version)
$ gem install bundler -v 2.2.17
$ bundle install
$ make serve
```

* 혹시나 yarn으로 이상으로 안된다면 추가

```bash
$ npm install -g yarn
$ make serve
```

* 로컬 빌드후에는 _site디렉토리에 결과를 보실수 있습니다

```bash
$ make build
```

* 혹은 `make serve` 후에 127.0.0.1:4000 에서 확인하기

## 2. 정기 미팅 (8/15) 20:00~

### 다음 주에 할 일

* 기존 문서 오류 찾기
* 조별토론(pr리뷰 or 화상미팅) - 조장님이 양 분배
* 리뷰어로 두 멘토님 초대

### 스펠링 체크 방법

* 에디터로 오타 찾기
  * 한글, 워드 등에서 파일 불러서 오타 찾기
  * 문서 하나하나 자세히 볼 때 사용하면 좋음
* hunspell로 오타 찾기
  * hunspell 및 hunspell-ko 설치
  * `hunspell -l index.html | sort | uniq` : 개별 파일 검사해보기
  * `find . -iname "*.html" -exec hunspell -l {} \;` : 디렉토리에서 여러 파일 검사해보기
* 용어집에 있는지 체크 후 없으면 용어집에 pr 올리기

## 3. 번역 연습 (~8/19까지)

### 파트 나누기

* [연습용 레포](https://github.com/hyoyoung/PyTorch-tutorials-kr-exercise)에서 pull requests로 번역을 해서 보내주세요.
  * beginner_source/saving_loading_models.py
  * beginner_source/text_sentiment_ngrams_tutorial.py
  * beginner_source/torchtext_translation.py
  * beginner_source/transfer_learning_tutorial.py
  * **beginner_source/translation_transformer.py** : 🙋‍♀️
* 방법
  * 각자 할 파일 선택해서, 파일 전체를 할 필요는 없고, 10~30줄 이내로 번역 후 PR 보내기
  * PR 보낼 때 리뷰어로 각 조 인원은 필수로 넣고, github id: hyoyoung, 9bow 를 필수로 추가하기

## 4. 조별 토론 (8/20) 20:00~

- 다른 사람이 올린 번역 PR 검토하기
- 용어집 및 기존 사례와 잘 어울리는지 확인하기

## 5. 정기 미팅 (8/22) 20:00~

* 이번 주 회고
  * 과제 의도 : 기존 번역된 문서와 비교하며 톤앤매너를 맞추는 것이 중요
  * 개선할 점
    * 주어는 번역을 잘 하지 않는다.
    * 처음 등장하는, 애매한 단어는 원문을 함께 쓰기도 한다.
    * class 이름과 일반 명사는 가급적 다르게 번역
    * 어디까지 자세하게 번역해야할지는 아직 토론 필요
* 다음 주부터 과제 두 개
  * 파이토치 튜토리얼에서 오타 잘못된 부분 찾아서 pr 실제로 찾아주기 (방법은 알아서, 최소 한 개 이상)
  * 용어집에서 오타 찾거나 추가하기 (merge 될만한 것들을 주로 올려주면 좋을 것 같다)
  * 개인주도적으로 활동하고, 토론하는 일이 많을 것
* 다음 달부터 마스터즈 프로그램
  * 목표 : 최소한의 적절한 가이드 + 오픈소스에 대한 열정
    * 좋은 기억을 가지고 갈 수 있는 컨트리뷰톤
    * 채워야 했던 숫자보다는 뿌듯한 기억 한 조각, 잘 번역한 문서가 기억나면 좋을 것 같다.
  * 실천하기
    * 조금만 더 의견 내고 참여하자
    * 모임 공간 등 이용해도 좋다

