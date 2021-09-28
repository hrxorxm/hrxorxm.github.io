---
layout: post
title:  "[파이토치 튜토리얼/허브 번역] 1. 튜토리얼 사이트 빌드 및 오픈소스 교육"
subtitle:   "오픈소스 컨트리뷰션 아카데미 1주차"
categories: "OpenSource-Contribution"
tags: []
use_math: true
---

# Week1

## 1. 튜토리얼 사이트 빌드하기 (8/9)

### [PyTorch 한국어 튜토리얼 기여하기](https://github.com/9bow/PyTorch-tutorials-kr/blob/master/CONTRIBUTING.md)

1. 해당 문제가 논의/진행 중인지 파악
2. ✨**이슈(issue)를 검색하거나 새로 남기기**✨
3. 저장소 복제(fork)하기
   * [[GitHub Docs] Fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
4. 로컬에서 빌드하기 (in 윈도우)
   * 환경 설정 : `pip install -r requirements.txt`
     * pip install 시 error: Microsoft Visual C++ 14.0 is required. : [Microsoft C++ Build Tools 다운로드](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)
   * 빌드 : `make html-noplot` 명령 후 `_build/html` 디렉토리에서 결과물 확인
     * 'make'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는 배치 파일이 아닙니다. : [윈도우(Windows)에서 make 명령 사용하기](https://ndb796.tistory.com/381)
5. 원본 경로/문서 찾기
6. 번역/수정하기
7. 로컬에서 결과 확인하기
   * `make html-noplot`
8. ✨**Pull Request 보내기**✨
   * [[GitHub Docs] About pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
   * [[GitHub Docs] Creating a pull request from a fork](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
9. Pull Request된 튜토리얼 문서 리뷰
   * [[GitHub Docs] About pull request reviews](https://docs.github.com/en/github/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)

### reStructuredText 문법 익히기

* [Quick reStructuredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html)
* [reStructuredText docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)

### 용어집 읽어보기

* [구글 머신러닝 용어집](https://developers.google.com/machine-learning/glossary?hl=ko)
* [TTA 정보통신 용어사전](https://terms.tta.or.kr/main.do)
* [파이토치 튜토리얼 번역 가이드 - 용어집](https://github.com/9bow/PyTorch-tutorials-kr/blob/master/TRANSLATION_GUIDE.md)

## 2. 오픈소스 기본 교육 (8/14) 13:00~17:00

### 오픈소스 개발참여준비

* 오픈소스의 GitHub 저장소에서 Fork
* Fork한 내 저장소를 Clone
* `git remote add upstream [원본 저장소 주소]` : 오픈소스 GitHub 저장소 설정

### 개발자가 오픈소스를 읽는 방법

* `git shortlog -sn -- mnist/ | nl` : 그 폴더에 기여를 많이 한 사람에게 리뷰를 부탁하면 된다.
* `git log --oneline` : commit log를 볼 수 있다.
* `git log --no-merges --oneline` : merge commit을 제외하고 볼 수 있다.
* 커밋 메세지 : 협업을 위해서 대충 적으면 안된다.
  * update 는 너무 추상적인 단어, remove, fix 등 다양한 단어 필요하다..
  * fix : 잘못된걸 고쳤다.
  * improve : 잘되던 기능을 더 잘되게 고쳤다.
  * add : 없던 기능을 추가하였다.
  * Correct typo : 오탈자 수정
* `git show [commitID] | grep "diff --git" | wc -l` : 한 커밋에서 몇개의 파일을 수정했는지 확인한다.
* `git log --oneline --after=2020-01-01 --before=2020-06-30`
  * 이 기간에 어떤 것이 주로 개발되었는지 알 수 있다.
  * 깃 프로젝트를 분석할 수 있는 reading skill 이다.
  * commit history가 대충 관리되어 있으면 의미가 없다.

### 오픈소스 개발 참여를 위한 수정 작업

* 순서
  * `git checkout -b [branchName]` : branch 만들기
  * 파일 수정 후 add, commit 하기
  * Fork한 저장소에 push 하기
  * Fork한 저장소의 작업 브랜치로 가서 Contribute - Open Pull-Request 제출하기
  * 관리자가 Merge
* Rebase
  * 상황 : 내가 수정한 것보다 다른 사람들이 수정한 것이 merge될 수 있다.
  * 해결 : Rebase (최신역사로 베이스를 갱신한다.)
    * `git fetch upstream master` : 저장소에서 최신 역사를 다운로드한다.
    * `git rebase upstream/master` : 위에서 가져온 것을 기반으로 rebase를 한다.
    * `git push --force origin fix-mnist` : 다시 force 푸시하면 자동으로 PR이 갱신된다.
* Stash
  * add하기 전에 stash하면 임시저장해놓을 수 있음
  * `git status` : 수정한 파일 확인
  * `git stash` : 임시저장
  * `git stash pop` : 임시저장 해둔 내용 복구
  * 또 다른 파일 복구 : `git checkout -- mnist/main.py`
    * Stash는 아예 날려버리고 싶지 않을 때 쓰고, checkout은 그냥 날려버리고 싶을 때 쓴다.
* Checkout
  * git의 history 창고로부터 뭔가를 가져올 때 쓰는 명령
    * branch를 가져오는 것도 마찬가지
    * 원래 파일을 최신 history를 기반으로 가져오는 것 (위 파일복구)
* 명령 취소
  * `git reset` : add 취소
  * `git reset --hard HEAD~1` : HEAD에서 commit 1개만 지운다.
* 커밋 수정
  * 일단 잘못 수정한 파일 수정해서 add 까지 하기
  * `git commit --amend` : (가장 최신) 커밋 자체를 수정 (commit ID 가 바뀐다.)
  * `git commit --amend -m "Commit message"` : 커밋 메시지도 수정
  * 리뷰를 받고 중간에 있는 커밋을 수정하고 쪼개는 작업이 많이 필요하다.
* 서명 넣기
  * 라이센스에 대한 규정을 다 이해하고 작업했다는 체크
  * `git commit -s -m "commit message"` : Signed-off-by
  * CLA(Contributor License Agreements) : 서명 안 넣고 하는 방법
    * ex) 구글의 오픈소스에 참여할 때는 [구글 CLA](https://cla.developers.google.com/about/google-individual) 이용

## 3. 오픈소스 심화 교육 (8/15) 13:00~17:00

### Rebase 실습

* `git rebase -i --root` : 수정 내역(commit) 과거시점으로 되감기(rewind)
  * 커밋 앞의 `pick` 을 `edit` 으로 수정해서 저장한다.
  * 여러개를 수정하면 여러번 continue로 풀면 된다.
  * 과거시점에서 commit을 추가할 수 있다.
    * 커밋을 또 잘못했으면,
    * `git reset --hard origin/master` : clone 받았을 때의 원본 master로 원상복구 된다.
  * 합치고 싶은 commit이 있으면, 그 중 최신 commit을 edit
    * `git reset --soft HEAD~1` : soft는 커밋을 지우긴 하지만 file을 남겨두고, hard를 파일도 날려버린다.
    * `git commit --amend` : 커밋을 수정한다(합친다.)
  * `git rebase -i HEAD~5` : rebase를 적용할 후보 범위를 지정할 수 있다.
* `git rebase --abort` : rebase 중에 rebase를 취소한다는 의미
* `git rebase --continue` : 수정내역(commit) 다시 현재 시점으로 풀기(continue)
* rebase 후에 commitID가 변경될 수 있다.

### Blame 실습

* Blame 을 적절히 쓰면서 소스리딩을 하면 분석할 때 더 도움이 된다.
* `git blame src/node.cc`
  * 이 소스코드가 왜 작성됐는지는 어떤 커밋에서 작성되었는지를 확인해보면서 더 잘 이해할 수 있다.
* 미션: node_http_parse.cc의 Parser 클래스 만든 최초 Commit 찾기
  * github에서 그 파일의 blame 을 보면 그 과거로 계속 돌려서 볼 수 있다.
  * `git log --oneline --reverse -- src/node_http_parser.cc | head -1` : 핵심 클래스이므로 파일이 생성됐을 때 같이 만들어졌을 가능성이 있다.
  * 또는 `{` 와 같은 것들은 거의 수정되지 않으므로 이것이 생성된 커밋 찾기
* 미션: node.cc 파일이 생성된 진짜 최초 Commit 찾기
  * 파일의 경로가 옮겨질 수 있다.
  * `git log --oneline --reverse -- src/node.cc | head -1` : 이 결과는 진짜 최초커밋이 아니라 그 폴더에서의 최초커밋이다.
  * `git reset --hard [위의 커밋ID]~1`
  * `git log --oneline --reverse -- node.cc | head -1` : 이 결과가 아마 진짜 최초커밋

### 좋은 커밋 메세지

* Commit Message Style Guide
  * 구조
    ```
    type : Subject
    
    body
    
    footer
    ```
  * 제목
    * Type : 카테고리 분류
      * **feat:** A new feature
      * **fix:** A bug fix
      * **docs:** Changes to documentation
      * **style:** Formatting, missing semi colons, etc; no code change
      * **refactor:** Refactoring production code
      * **test:** Adding tests, refactoring test; no production code change
      * **chore:** Updating build tasks, package manager configs, etc; no production code change
    * Subject : 한줄요약
      * no greater than 50 characters
      * begin with a capital letter
      * Use an imperative tone : (ex) use **change**; not changed or changes.
    * (예시) `fs ext4: Fix the B problem`
      * 보통 why: 20%  how: 80% 인 경우가 많다. 근데 how는 코드를 보면 알 수 있다.
      * why: 80% how: 20% 가 적당하다. 근데 how가 보편적이지 않은 방법이면 써주는 것이 중요할 때도 있음
  * Body
    * 구체적인 서술
    * 본문은 `어떻게`보다 `무엇을`, `왜`에 맞춰 작성하기
  * Footer
    * 참고할 commitID나 해결한 issue번호
    * 해당 line에 대한 blame 조사와 cc 등 명시
  * 라이센스와 본인인증
    * signed-off-by
    * CLA 등록
  * 예시
    * [Linus Tovalds가 설명한 좋은 Commit Message](https://github.com/torvalds/subsurface-for-dirk/blob/a48494d2fbed58c751e9b7e8fbff88582f9b2d02/README#L88)
    * [Commit History를 효과적으로 관리하기 위한 규약: Conventional Commits](https://medium.com/hdackorea/commit-history%EB%A5%BC-%ED%9A%A8%EA%B3%BC%EC%A0%81%EC%9C%BC%EB%A1%9C-%EA%B4%80%EB%A6%AC%ED%95%98%EA%B8%B0-%EC%9C%84%ED%95%9C-%EA%B7%9C%EC%95%BD-conventional-commits-67b2114ac8e4)

