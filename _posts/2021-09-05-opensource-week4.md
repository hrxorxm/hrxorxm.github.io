---
layout: post
title:  "[파이토치 튜토리얼/허브 번역] 4. 중간 보고서 제출 및 PR 보내는 방법 논의"
subtitle:   "오픈소스 컨트리뷰션 아카데미 4주차"
categories: "OpenSource-Contribution"
tags: [OpenSource, Contribution, Academy]
use_math: true
---

# Week4

## 1. 후원사 기념품 언박싱 (8/31)

* 후원사 : Microsoft, LINE, LG전자, SKT, Kakao, Lablup, GitHub, NexCloud
* 💌기념품 사진💌
  ![image](https://user-images.githubusercontent.com/35680202/131505706-ea51025f-a327-4e8a-b838-7ce4ed428dbc.png)

## 2. 중간 보고서 제출 (~9/1까지)

* 공지
  * 8월 7일 발대식 기준으로 시작된 Challenges 컨트리뷰션 프로그램이 31일 기준으로 종료됩니다.
  * 종료 일정에 맞춰 멘티와 멘토 모두의 중간보고서 작성 일정과 절차를 사전에 안내드립니다.
* 방법
  * 멘티 개인별 노션 페이지를 생성해 편집가능한 유저로 초청해 드릴 예정입니다.
  * 개별보고서는 멘티 개인에게 고유 액세스를 드린 페이지로 링크를 전달해도 다른 멘티가 조회할 수 없습니다.
  * 보고서 작성 권한은 8월 27일부터 9월 1일까지 유효합니다. (이후 일괄 보기권한 전환)

## 3. 문서 번역 기여하기

* 방법 : 이번 주부터 자유롭게 문서 선정해서 번역 기여하면 된다.
* 주제 : [`beginner/audio_preprocessing_tutorial.py` 목차 하이라이팅 오류](https://github.com/9bow/PyTorch-tutorials-kr/issues/312)
* 내용
  * 오디오 섹션 내의 일부 문서들 간의 이동 시, 목차 하이라이팅이 제대로 반영되지 않는 문제 발견
  * 원본 사이트에서도 똑같은 현상이 있기 때문에, 원본 사이트에 다시 이슈를 남기기로 결정

## 4. 조별 미팅 (9/3) 20:00~

* 이번 주 조별 토론 주제 : 번역 및 기여 가이드 확장 (번역 스타일링 및 기여 방법에 대해서 토론 및 정리)
  * a조 : 번역 스타일링
  * b조 : 자주 틀리는 실수 및 reStructure 문법
  * **c조 : pr 보내는 방법**
* pr 보내는 방법 정리
  * 이슈 남기기
  * 빌드
    * docker container를 이용한 환경셋팅 제안
  * PR 보내기
    * Github action을 이용하여 build 실패 시 auto reject할 수 있는 기능 제안 ([예시](https://github.com/LUXROBO/pymodi/blob/master/.github/workflows/build.yml))
  * 피드백 반영하기
    * comment rule 제안
    * Gist를 사용하여 댓글을 이용한 논의 제안
    * Github Kanban board 제안

## 5. 정기 미팅 (9/5) 20:00~

* 다음 주부터 마스터즈 프로그램, 리드멘티 선정
* 조별 발표
  * a조 : 번역 스타일링
    * 번역 스타일링
      * 원본 내용을 자연스럽게 번역하면 되는 것이기 때문에 타이트한건 정하지 않음
      * you, we 는 해석하지 않는다.
      * 어느 정도까지 괄호치고 얘기할지 : best guess 처럼 일반적인 용어는 원본을 표시하지 않아도 될 것 같다. 애매한건 투표로 결정하자
    * 말투
      * 경어체 : 이야기 하듯이 / 평어체 : 정보 전달 어체
      * 원어를 쓰는 것도 작성자마다 글 쓰는 방식이 다르다.
      * 번역자가 한 문서 안에서라도 일관된 톤을 유지하기만 해도 좋겠다.
      * => 멘토님 코멘트 : 최소 문서 단위에서는 통일, 그게 아니라면 가이드라인 제시, 생각을 더 해보고 문서화하자
    * 그 외
      * 공식 튜토리얼 사이트에 파비콘이 없는데 추가하면 어떨까
      * 한국어 번역 링크에서 원어 링크로 연결하는 것도 있으면 좋겠다.
  * b조 : 자주 틀리는 실수, restructuredText 문법
    * 자주 틀리는 실수
      * 브랜치를 생성 안하고 pr 생성했을 때
      * 컴파일을 안하고 pr 했을 때 글자가 깨지는 경우
      * pr 올릴 때 용어집대로 안하는 경우
      * 줄내림과 공백을 원문대로 지키지 않는 경우
    * 문법
      * 글자수에 맞게 = 기호 넣어야 하는 것
      * 하이퍼링크나 인덴트 문법
  * c조 : pr 보내는 방법
    * 환경설정 : 도커 만들면 어떨까
    * pr 날리기 전에 빌드를 강제하기 위해서 before after 사진 첨부나 github action을 이용하자
      * => 멘토님 코멘트 : github action 은 도입을 고려할만 한 것 같다.
    * comment : 수정하면 내역이 사라져서 반영 여부를 확인하는게 어려워질 수 있다. 논의가 길어질수록 공유할 필요성이 생긴다.
* 용어집 투표 : 10명 이상 넘으면 approve
  1. **training - 학습**
  2. bias - 편향
  3. fork - 복제
  4. fully-connected layer - fully-connected 계층
  5. transform, transformation - 변환
  6. label - 라벨
* 질의응답
  * 허브 사이트 번역 시작해도 되는데, 아직 이슈, PR 템플릿이 없어서 튜토리얼 때 했던 방식으로 올려주면 된다.
  * 튜토리얼에서 번역해야하는 남은 문서들은 한 문서가 굉장히 긴데, 번역이 다 완료되지 않더라도 진행상황을 나누면서 하고싶다면 pr을 [draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/)형식으로 올릴수도 있다.

