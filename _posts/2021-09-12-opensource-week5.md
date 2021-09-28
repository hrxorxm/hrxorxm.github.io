---
layout: post
title:  "[파이토치 튜토리얼/허브 번역] 5. 중간 공유회 및 허브 번역 시작"
subtitle:   "오픈소스 컨트리뷰션 아카데미 5주차"
categories: "OpenSource-Contribution"
tags: []
use_math: true
---

# Week5

## 1. 중간 공유회 준비 (9/7) 21:30~

* 리드멘티 선정 완료! 마스터즈 프로그램 시작!
* 중간 공유회 (Midterm Recap)
  * 내용 : 중간 공유회 (Recap) 발표 5분 정도
  * 분량 : 표지 1장, 내용 3장
  * 발표 : 깃헙 pr, 오타 자동화, 용어집 추가 확장한 내용 등 정리해서 준비하기

## 2. 허브 번역 기여 시작하기

* 이슈 등록 : [huggingface_pytorch-transformers.md 번역 예정 #1](https://github.com/9bow/PyTorch-hub-kr/issues/1)
* 환경 셋팅 : [WSL에서 vscode 사용하기](https://docs.microsoft.com/ko-kr/windows/wsl/tutorials/wsl-vscode)
* ✅ 중요 : 빌드는 [PyTorchKR 레포](https://github.com/9bow/PyTorchKR)로 하기!

## 3. 중간 공유회 (9/11) 13:00~

* 발표 시간 : 파이토치 튜토리얼/허브 번역 - 13:20~13:25

## 4. 조별 미팅 (9/11) 20:00~

* 목표 : pr 보내는 방법을 howto 형태로 정리, fork부터 pr 보내기, 리뷰는 이렇게 해주세요 등 세부 주제 위주 내용 정리
* 기한 : 토요일까지 (9/11)
* 방법 : 번역 스타일링 및 기여 방법 문서(markdown 포맷)를 채널 슬랙으로 보내주세요.

## 5. 정기 미팅 (9/12) 20:00~

### 이번 주 공지

* 중간 공유회에서 리눅스 커널 커밋이 인상적!
* 같은 조에서 번역한 내용은 꼭 리뷰해주기

### 조별 발표

* a조 : 스타일링 관련 정리
  * 글꼴 : 슬랙에서 투표 진행 중, Noto Sans KR 이 유력
  * 종결 어미 : `~다.` 권장 `~요` 도 허용
  * 기계적인 번역보다는 의역을 하는 것을 우선시
  * 문서화 진행할 때 github discussion을 활용해서 문서화를 진행하는 것이 어떨까?
  * Github Action : 빌드 자동화, PR 라벨 달기(label action)
* b조 : 자주 틀릴만한 부분
  * 자주 틀리는 실수 : 기본적인 PR 방법 등
  * reStructuredText 문법 관련 내용 정리
  * 번역 스타일링 관련 :  주어 생략, 문장이 길어지면 줄바꿈해주기, 애매한 단어는 원문과 함께 사용 등
* c조 : PR 보내는 방법 정리

### 용어집

* 확장이 필요하긴 한데, 논의가 필요한 단어들이 많아 더 고민해보고 결정

