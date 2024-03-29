---
layout: post
title:  "[Boostcamp AI Tech] 8주차 - AI 엔지니어로서 알면 좋을 지식들"
subtitle:   "Naver Boostcamp AI Tech Special Lecture"
categories: "Boostcamp-AI-Tech"
tags: [Special-Lectures]
use_math: true
---

# 부스트캠프 8주차

## [1] 서비스 향 AI 모델 개발하기 - 이활석 (Upstage)

* 연구 관점에서의 AI 개발 : 정해진 데이터셋/평가 방식에서 더 좋은 모델을 찾는 일
* 서비스 관점에서 AI 개발 : 학습 데이터셋, 테스트 데이터셋, 테스트 방법도 없다. 주어지는 것은 서비스 요구 사항만 있다.
* 학습 데이터셋 준비 : 서비스 요구 사항으로부터 학습 데이터셋의 종류, 수량, 정답에 대해 정해야 한다.
  * 종류 : 서비스 기획팀과 의사소통을 통해, 각각 몇 장을 수집할 것인지 정해야한다.
  * 기술 모듈 설계 : 원하는 입출력을 갖는 기술 모듈 개발
  * 정답 : AI 모델 별로 입력에 대한 출력 쌍 정하기
  * 학습 데이터 준비와 모델 파이프라인 설계 과정이 반복되며 수렴된다.
* 테스트 데이터셋 준비 : 학습 데이터셋에서 일부 사용 (아닌 경우도 많음)
* 테스트 방법 준비 : 서비스 요구사항으로부터 도출, 오프라인 테스트 결과가 온라인 테스트 결과와 유사하게 오프라인 테스트를 잘 설계해야 한다.

  |           | Offline 서비스 적용 전 성능 평가                             | Online 서비스 적용 시 성능 평가                          |
  | --------- | ------------------------------------------------------------ | -------------------------------------------------------- |
  | 정량 평가 | 완벽하지 않기 때문에 AI 모델 **후보 선택 목적**으로 활용     | 해당 AI 모델을 서비스 시나리오에서 **자동 정량 평가**    |
  | 정성 평가 | 각 후보 AI 모델에 대한 **면밀 분석** 후 서비스 출시 버전 선택 | **VOC**(Voice Of Customer)<br />AI 모델 개선 포인트 파악 |

* 모델 요구사항 도출
  * 처리 시간 : 하나의 입력이 처리되어 출력이 나올 때까지의 시간
  * 목표 정확도 : 해당 기술 모듈의 정량적인 정확도
  * 목표 QPS(Queries Per Second) : 초당 처리 가능한 요청 수
  * Serving 방식 : 모바일에서 동작하는지, CPU/GPU, 로컬/클라우드
  * 장비 사양 : 예산, QPS에 맞춰서 정하기
* 서비스 향 AI 모델 개발 기술팀의 조직 구성
  * AI 모델팀 : 서비스 요구사항 → AI 모델
    * Modeling - Modeler : AI 모델 구조 제안, AI 모델 성능 분석/디버깅
    * Data - Data Curator : 학습데이터 준비(외주 업체 대응), 정량 평가 수립, 정성 평가 분석
    * Tool - IDE Developer : 라벨링 툴 개발, 모델 분석 툴 개발, 모델 개발 자동화 파이프라인 개발
    * Model Quality Manager : 모델 품질 매니저 (총괄 매니저)
  * AI 모델 서빙팀
    * Model Engineering - Model Engineer : 모바일/GPU에서의 구동을 위한 모델 변환, 경량화(Lightweight) 작업, GPU 고속처리(CUDA programming), C++/C로 변환 등
    * App Developer / BE Engineer 등
* 조언
  * 개발자 → AI 관련 전환 : Model Engineering / Tool / Serving 은 개발력이 많이 필요한 일이고 니즈가 많아지기 때문에 한번에 AI 모델링 쪽으로 넘어가지 않아도 될 것 같다.
  * 모델러 : AutoML 등 점점 업무가 자동화되고 인력 수도 늘고 있기 때문에 주변으로 역량을 확대하면 좋을 것 같다. (AI 기술 분야 확장, FE 개발 능력 확장, BE 개발 능력 확장 등)
  * AI 기술트렌드에 민감해야 한다. 자기만의 노하우 만들기

## [2] 캐글 그랜드마스터의 노하우 대방출 - 김상훈 (Upstage)

* 파이프라인의 빠르고 효율적인 반복
  * GPU 장비 : google colab은 보조로 사용
  * 본인만의 기본 코드 만들기
* 점수 개선 아이디어
  * Notebooks / Discussion 탭 참고하기
  * 대회 마지막 제출 때까지 점수 개선 아이디어 고민하기
* 탄탄한 검증 전략
  * 최종 순위 하락을 피하기 위해 필요
  * 일반화 성능이 높은 모델이 좋은 모델
  * 점수 갭을 줄이기 위한 (stratified) k-fold 검증 전략 구축하기
* 기타 꿀팁
  * 앙상블 : 모델별 예측값을 섞는 것, weighted sum의 방법을 많이 사용함
  * 앙상블 전에 높은 점수의 싱글 모델이 필요함

## [3] AI + ML과 Quant Trading - 구종만 (Tower Research Capital)

* 투자(investment) vs 트레이딩(trading) : 트레이딩은 상대적으로 단기간 (최대 3일정도)
* 퀀트 트레이딩
  * Quantitative(계량적) 트레이딩
  * 모델 기반(가격이 특정 수학적 성질을 가진다고 가정) 혹은 데이터 기반(시장의 과거 데이터에서 분포를 추정) 접근
  * Automated / system / algorithmic trading
* 예시
  * arbitrage : 싼 곳에서 사서 비싼 곳에서 판다. (같은 상품의 가격을 맞춰주는 역할) → 90% 속도(1~2초) + 10% 알파 경쟁
  * market making : 매수 주문과 매수 주문을 동시에 낸다. 두 주문의 가격차만큼 이득, 유동성 공급 → 50% 속도(30초~1분) + 50% 알파 경쟁
  * statistical arbitrage : 미래 가격의 변화를 예측해서 거래한다. 데이터 기반 접근이 필수적 → 10% 속도 + 90% 알파
* 퀀트 트레이딩 플레이어들
  * 퀀트 헤지펀드 / 로보 어드바이저 : 고객의 자본 운영
  * 프랍 트레이딩(자기 자본 거래) : 회사 파트너들의 자본을 거래
  * 금융위기 이후 규제 변경으로 은행들은 자기 자본 거래를 하지 않음

## [4] 내가 만든 AI 모델은 합법일까, 불법일까 - 문지형 (Upstage)

* AI와 저작권법
  * 아직 저작권법은 AI 모델 개발을 고려하지 않은 부분이 많다.
  * 저작권 : 사람의 생각이나 감정을 표현한 결과물(저작물)에 대하여 창작자에게 주는 권리 → **창작성**이 있다면 별도의 등록절차없이 자연히 발생한다.
  * 저작물 : 사람의 생각이나 감정을 표현한 결과물
  * 저작권법에 의해 보호받지 못하는 저작물 : 법률, 사실 전달에 불과한 시사보도 등
* 합법적으로 데이터 사용하기
  * 저작자와 협의
    * 저작재산권 독점적/비독점적 이용허락
    * 저작재산권 전부/일부에 대한 양도
  * 저작자가 명시한 **라이센스**에 맞춰서 이용하기
    * CCL(Creative Commons License)
      * BY : Attribution - 적절한 출처와 저작자 표시
      * ND : NoDerivatives - 변경 금지
      * NC : NonCommercial - 비영리
      * SA : ShareAlike - 동일조건 변경허락
  * 예시
    * 나무위키 데이터를 크롤링해서 MRC 데이터셋 제작을 한 후에 깃헙을 통해 배포하는 것이 가능한지 → 학교 소속이라면 비영리목적으로 간주되므로 가능, 원 데이터의 라이센스인 CC-BY-NC-SA를 부착하고, 원 데이터의 출처 명시해야함
    * KorQuAD의 질문만 바꿔서 새롭게 MRC 데이터셋을 제작한 이후에 깃헙에 배포해도 되는지 → CC-BY-ND 조건에서 변경금지 조건이 걸려있기 때문에, KorQuAD의 지문, 질문, 정답쌍을 변경하여 공개하는 것은 불가능함
    * 뉴스 데이터 이용 : 뉴스 기사의 저작권은 언론사에 있음, 한국언론진흥재단에서 대부분의 언론사의 저작권을 위탁해서 관리함, 한국언론진흥재단 또는 직접 언론사에 문의 (간혹 CCL 적용된 경우 있음), 뉴스 기사의 제목은 저작권법의 보호를 받지 못함
    * 공정 이용(Fair-use) 에 대해서는 저작권자의 허락을 받지 않고도 저작물 이용 가능 : 교육, 학교 교육 목적 등

## [5] Full Stack ML Engineer - 이준엽 (Upstage)

* Full Stack ML Engineer
  * 정의
    * ML Engineer : 머신러닝 기술을 이해하고, 연구하고, Product를 만드는 Engineer
    * Full Stack Engineer : 모든 스택을 잘 다루려는 **방향**으로 가고 있는 사람, 내가 만들고 싶은 Product를 시간만 있다면 모두 혼자 만들 수 있는 개발자
    * **Full Stack ML Engineer** : Deep learning research를 이해하고 + ML Product로 만들 수 있는 Engineer
  * 장점
    * 처음부터 끝까지 내 손으로 만들 수 있어서 재밌다.
    * ML 모델의 빠른 프로토타이핑
    * 연결에 대한 고려가 들어간 개발 결과물
    * 각 스택의 깊은 이해에 도움 (기술에 대한 이해 == 잠재적 위험에 대한 고려)
    * 성장의 다각화
  * 단점
    * 깊이가 없어질 수도 있다. (모든 스택에서 최신 트렌드를 따라잡기 어려운게 당연함)
    * 절대적으로 시간이 많이 들어간다.
* Full Stack ML Engineer in ML Team
  * ML Product 단계 : 요구사항 전달 → 데이터 수집 → ML 모델 개발 → 실 서버 배포
  * 실생활 문제를 ML 문제로 Formulation
  * Raw Data 수집 (`Selenium` 등 이용)
  * Annotation tool 개발 (`Vue.js`, `django`, `MySQL`, `docker` 등 이용)
  * Data version 관리 및 loader 개발 (`S3` 등 이용)
  * Model 개발 및 논문 작성 (`Pytorch`, `Tensorflow` 등 이용)
  * Evaluation tool 혹은 Demo 개발 (`Flask`, `Angular`, `D3.js` 등 이용)
  * 모델 실 서버 배포
* Roadmap to Full Stack ML Engineer
  * Stackshare
    * Frontend : `Vue.js`, `Angular`
    * Backend : `django`, `Flask`, `ruby on rails`
    * Machine Learning : `PyTorch`, `TensorFlow`
    * Database : `MySQL`, `Marua DB`, `redis`, `amazon DynamoDB`
    * 그 외 : `docker`, `git/github`, `aws`
  * 각 스택에서 점점 framework의 inferface가 쉬워지는 방향으로 발전하고 있다.
  * 익숙한 언어 + 가장 적은 기능 + 가장 쉬운 Framework로 시작하자
  * 처음부터 너무 잘 만들려고 하지 말고, 최대한 빨리 완성해보자
  * 배우고 싶었던 스택에 대한 문서나 유튜브부터 재미로 보자
  * 만들고 싶은 것이 없다면, 하나의 논문을 구현하고 Demo page를 만들어보는 것을 추천!

## [6] AI Ethics - 오혜연 (KAIST)

* AI & Individuals
  * Bias Source - [Big Data's Disparate Impact](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899)
  * Privacy - [Singapore TraceTogether App](https://arxiv.org/abs/2003.11511)
* AI & Society
  * Social Inequality - [The AI Now Report](https://ainowinstitute.org/AI_Now_2016_Report.pdf)
  * Misinformation - Deepfakes
* AI & Humanity
  * [AI for Health](https://www.microsoft.com/en-us/ai/ai-for-health)
    * [AI model improves breast cancer detection on mammograms](https://youtu.be/Mur70YjInmI)
    * [Improving HIV care for teens in Kenya with virtual friendships](https://www.path.org/articles/improving-hiv-care-for-teens-in-kenya-with-virtual-friendships/)
    * [Artificial intelligence–enabled rapid diagnosis of patients with COVID-19](https://www.nature.com/articles/s41591-020-0931-3)
  * AI for Climate Change
    * ⚡ [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/10.1145/3442188.3445922)
    * [Tackling Climate Change with Machine Learning](https://arxiv.org/abs/1906.05433)
    * [Gigawatt-hour scale savings on a budget of zero: Deep reinforcement learning based optimal control of hot water systems](https://www.sciencedirect.com/science/article/abs/pii/S0360544217320388)
    * [Sequences of purchases in credit card data reveal lifestyles in urban populations](https://www.nature.com/articles/s41467-018-05690-8)
    * [DeepMind Uses AI to Reduce Energy Use](https://deepmind.com/blog/article/deepmind-ai-reduces-google-data-centre-cooling-bill-40)

## [7] AI 시대의 커리어 빌딩 - 박은정 (Upstage)

* Careers in AI
  * AI를 다루는 회사의 종류
    * AI for X : AI로 기존 비즈니스를 더 잘 하려는 회사 (AI는 보조수단, 대부분의 회사)
    * AI centric : AI로 새로운 비즈니스를 창출하려는 회사 (신생 회사가 많음)
  * AI를 다루는 팀의 구성
    * Business : 사업 기획자, 서비스 기획자, 마케팅/세일즈/PR, 법무/윤리학자
    * Engineering : Data engineer, AIOps engineer, AI engineer, AI researcher, AI manager
* How to start my AI engineering career
  * 시작하기
    * AI competition (예 : 캐글)
    * 최신 논문 재현 (참고 : [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021))
  * 공통적으로 필요한 역량
    * 컴퓨터 공학에 대한 기본적인 이해와 소프트웨어 엔지니어링 능력
    * 최신 기술을 빠르게 습득하기 위한 영어 독해 능력
    * Grit(끈기있는 자세), Humility(겸손함), Passion(열정), Teamwork(협력), Kindness(선함)
  * 이력서 : 강력한 한 방 (규모가 큰 + AI 관련)
    * Coding competitions
    * AI conpetitions
    * Publication record
    * 서비스 경험
    * 다른 회사 경력
  * 조언
    * 모집 공고 꼼꼼히 보기
    * 내가 어디에 강점을 가지는지 잘 알고, 엣지를 살릴 수 있는 포지션을 찾는 것이 중요!

## [8] 자연어 처리를 위한 언어 모델의 학습과 평가 - 박성준 (Upstage)

* 언어 모델링(Language Modeling)
  * 언어 모델링(Language Modeling) : 주어진 문맥을 활용해 다음에 나타날 단어 예측하기
    * **SQuAD** : 주어진 질문에 대해서 답 찾기 (질의응답)
    * SNLI : 두 문장 사이에 모순이 있는지 없는지 찾기
    * SRL : 의미적 역할 레이블링
    * Coref : 문단 내에서 무엇을 지칭하는지 찾기
    * NER : 문단 내 각 단어가 사람인지, 기관명인지 등을 맞추는 개체명 인식
    * SST-5 : 문장 분류
  * 양방향 언어 모델링(Bidirectional Language Modeling)
    * ELMo의 등장! 위 태스크들 모두를 잘 할 수 있게 만드는 것이 가능해짐
    * BERT 이후로, 사전학습 모델을 이용하여 파인튜닝하는 것이 일반적
* 언어 모델의 평가
  * GLUE 벤치마크 (Genetal Language Understanding Evaluation)
    * BERT 에서 사용한 평가
      * QQP(Quora Question Pairs) : 문장 유사도 평가
      * QNLI(Question NLI) : 자연어 추론
      * SST(The Stanford Sentiment Treebank) : 감성 분석
      * CoLA(The Corpus of Linguistic Acceptability) : 언어 수용성
      * STS-B(Semantic Textual Similarity Benchmark) : 문장 유사도 평가
      * MRPC(Microsoft Research Paraphrase Corpus) : 문장 유사도 평가
      * RTE(Recognizing Textual Entailment) : 자연어 추론
    * BERT 이후에 포함된 평가
      * MultiNLI Matched : 자연어 추론
      * MultiNLI Mismatched : 자연어 추론
      * Winograd NLI : 자연어 추론
  * 다국어 벤치마크의 등장 : FLUE(프랑스어), CLUE(중국어) 등
  * ✨**KLUE(Korean Language Understanding Evaluation)**✨ : 한국어 자연어 이해 벤치마크
    * 개체명 인식(Named Entity Recognition)
    * 품사 태깅 및 의존 구문 분석(POS tagging + Dependency Parsing)
    * 문장 분류(Text classification)
    * 자연어 추론(Natural Language Inference)
    * 문장 유사도(Semantic Textual Similarity)
    * 관계 추출(Relation Extraction)
    * 질의 응답(Question & Answering)
    * 목적형 대화(Task-oriented Dialogue)
