---
layout: post
title:  "[파이토치 튜토리얼/허브 번역] 2. 허브 사이트 빌드 및 번역 연습"
subtitle:   "오픈소스 컨트리뷰션 아카데미 2주차"
categories: "Contribution"
tags: [OpenSource, Contribution, Academy]
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
```
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
```
$ npm install -g yarn
$ make serve
```
* 로컬 빌드후에는 _site디렉토리에 결과를 보실수 있습니다
```
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

