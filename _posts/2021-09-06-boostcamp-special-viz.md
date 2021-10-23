---
layout: post
title:  "[Boostcamp AI Tech][Special] Data Viz"
subtitle:   "Naver Boostcamp AI Tech Special Lecture"
categories: "Boostcamp-AI-Tech"
tags: [Special-Lectures]
use_math: true
---

# 부스트캠프 데이터 시각화 강의

## [1강] Introduction to Visualization (OT)

* 시각화
  * 구성요소 : 목적, 독자, 데이터, 스토리, 방법, 디자인
  * 목표 : 모범 사례를 통해 좋은 시각화를 만들어보자
* `데이터` 시각화
  * 데이터셋 종류 : 정형, 시계열, 지리, 관계형, 계층적, 비정형 데이터
  * 수치형(numerical) : 연속형(continuous) / 이산형(discrete)
  * 범주형(categorical) : 명목형(nominal) / 순서형(ordinal)
* 시각화의 요소
  * 마크(mark) : 점, 선, 면
  * 채널(channel) : 마크를 변경할 수 있는 요소들
    * 위치, 색, 모양, 크기, 부피, 각도 등
    * 전주의적 속성(Pre-attentive Attribute)
      * 주의를 주지 않아도 인지하게 되는 요소
      * 적절하게 사용할 때, 시각적 분리(visual pop-out)
* `Matplotlib` : `numpy`와 `scipy`를 베이스로 하여, 다양한 라이브러리와 호환성이 좋다.

## [2강] 기본 차트의 사용

* Bar plot : 막대 그래프, 범주에 따른 수치값 비교에 적절
  * Principle of Proportion ink : 실제 값과 그래픽으로 표현되는 잉크 양은 비례해야 함
* Line plot : 꺾은선 그래프, 시간/순서에 대한 추세(시계열 데이터)에 적합
  * 보간 : 점과 점 사이에 데이터가 없을 때 잇는 방법
* Scatter plot : 산점도 그래프, 두 feature간의 상관 관계 파악에 용이
  * 인과 관계(causal relation)과 상관 관계(correlation)은 다르다.

## [3강] 차트의 요소

* Text 사용하기
  * 시각적으로만 표현이 불가능한 내용 설명
* Color 사용하기
  * 효과적인 구분, 색조합 중요, 인사이트 전달 중요
  * 범주형 : 독립된 색상 / 연속형 : 단일 색조의 연속적인 색상 / 발산형 : 상반된 색
  * 색각 이상(색맹, 색약) 고려가 필요할 수 있다.
* Facet(분할) 사용하기
  * Multiple View : 다양한 관점 전달
  * subplot, grid spec 등 이용

## [4강] 통계와 차트

* Seaborn : Matplotlib 기반의 통계 시각화 라이브러리, 쉬운 문법과 깔끔한 디자인
* Seaborn 기초 : 기본 차트 그리기
  * Categorical API : 범주형 데이터에 대한 시각화
    * 대표적 : `counterplot` / `boxplot` / `violinplot`
    * ETC : `boxenplot` / `swarmplot` / `stripplot`
  * Distribution API : 분포에 대한 시각화
    * Univariate Distribution : 단일 변수에 대한 분포
      * `histplot` : 히스토그램
      * `kdeplot` : Kernel Density Estimate
      * `ecdfplot` : 누적 밀도 함수
      * `rugplot` : 선을 사용한 밀도함수
    * Bivariate Distribution : 두 변수에 대한 결합 확률 분포(joint probability distribution)
      * `histplot`과 `kdeplot` 사용 시 축 두개 입력
  * Relational API : 관계에 대한 시각화
    * `scatterplot` / `lineplot`
  * Regression API : 회귀에 대한 시각화
    * `regplot`
  * Matrix API : Matrix 데이터에 대한 시각화
    * `heatmap(data.corr())` : 주로 상관관계(correlation) 시각화에 많이 사용
* Seaborn 심화 : 여러 차트 그리기
  * `jointplot` : 두 변수의 결합확률 분포와 함께 각각의 분포도 살필 수 있는 시각화를 제공
  * `pairplot` : 데이터셋의 pair-wise 관계를 시각화하는 함수, 서브플롯 100개 이하로 그리는 것 추천
  * FaceGrid : pairplot으로 feature-feature 사이를 살폈다면, Facet Grid는 feature-feature 뿐만이 아니라 feature's category-feature's category의 관계도 살펴볼 수 있다.
    * `catplot` : Categorical
    * `displot` : Distribution
    * `relplot` : Relational
    * `lmplot` : Regression

## [5강] 다양한 시각화 방법론

* Polar Coordinate (극 좌표계)
  * 구현 : [Matplotlib Projection](https://matplotlib.org/stable/api/projections_api.html)
  * Polar Plot
    * 극 좌표계를 사용하는 시각
    * 회전, 주기성 등을 표현하기에 적합
    * Line, Bar, Scatter 모두 가능
    * Matplotlib 공식 로고도 그릴 수 있다.
  * [Radar Plot](https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)
    * 극 좌표계를 사용하는 대표적인 차트
    * 중심점을 기준으로 N개의 변수 값 표현하기에 적합 (데이터의 Quality)
    * 각 feature는 독립적이며, 척도가 같아야 한다.
* Pie Charts
  * 구현 : [Pie Chart in Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html#matplotlib.pyplot.pie)
  * Pie Chart
    * 원을 부채꼴로 분할하여 표현하는 통계 차트
    * 전체를 백분위로 나타낼 때 유용
    * 비교가 어렵기 때문에 bar plot이 더 유용함
    * [Multiple Pie Charts: Unreadable, Inefficient, and Over-Used](https://www.readcube.com/articles/10.3138%2Fjsp.46.3.05)
  * Donut Chart
    * 중간이 비어있는 Pie Chart
    * 데이터 분석 EDA 에 있어서는 비추천
    * 인포그래픽, 스토리텔링 시에 종종 사용
  * Surburst Chart
    * 햇살(sunburst)을 닮은 차트
    * 계층적 데이터를 시각화하는데 사용
    * 가독성이 떨어지기 때문에 Treemap 추천
* 다양한 시각화 라이브러리
  * [missingno](https://github.com/ResidentMario/missingno)
    * 결측치(missing value)를 체크하는 시각화 라이브러리
  * [squarify](https://github.com/laserson/squarify)
    * Treemap : 계층적 데이터를 직사각형을 사용하여 포함 관계를 표현한 방법
    * Mosaic plot과 유사
    * [History of Treeviews](http://www.cs.umd.edu/hcil/treemap-history/index.shtml)
  * [PyWaffle](https://pywaffle.readthedocs.io/en/latest/)
    * Waffle Chart : 와플 형태로 discrete하게 값을 나타내는 차트
    * Pictogram Chart, 인포그래픽에서 유용
  * [pyvenn](https://github.com/konstantint/matplotlib-venn)
    * Venn : 집합(set) 등에서 사용하는 익숙한 벤 다이어그램
    * EDA 보다는 출판 및 프레젠테이션에 사용

## [6강] 인터랙티브 시각화

- Interactive를 사용하는 이유 : 정적 시각화는 원하는 메세지를 압축해서 담는다는 장점이 있지만, 공간적 낭비가 크다.
  - jupyter notebook : 함수형으로 1번에 1개를 만드는 방식
  - web demo : 모든 내용을 한 번에 db에 올려서 사용 (느릴 수 있음)
- 인터랙티브의 종류
  - Select, Explore, Reconfigure, Encode, Abstract, Filter, Connect 등
  - [Toward a Deeper Understanding of the Role of Interaction in Information Visualization](https://www.cc.gatech.edu/~stasko/papers/infovis07-interaction.pdf)
- 인터랙티브 시각화 라이브러리
  - [Plotly](https://plotly.com/python/)
    - 인터랙티브 시각화에 가장 많이 사용, 예시+문서화 강점
    - 통계, 지리, 3D, 금융 시각화 등
  - [Plotly Express](https://plotly.com/python-api-reference/plotly.express.html)
    - Ploty를 seaborn과 유사하게 만든 쉬운 문법
  - [Bokeh](https://docs.bokeh.org/en/latest/index.html)
    - 기본 테마가 Plotly에 비해 깔끔
    - 비교적 부족한 문서화
  - [Altair](https://altair-viz.github.io/)
    - Vega 라이브리를 사용하여 만들었고, 문법이 Pythonic하지 않다.
    - 기본 차트(Bar, Line, Scatter, Histogram)에 특화

## [7강] 주제별 시각화와 사용법

* 비정형 데이터 시각화 방법
  * Dataset Meta Data Visualization
    - Target Distribution : 훈련 상에서 발생할 수 있는 문제점 예측 (데이터 불균형 등 파악) → Augmentation 방법론 및 모델 선택 시 도움
  * Dataset Listup
    - Only Dataset : 데이터셋의 일부를 리스트업하여 확인
    - Datset-Target(bounding box 등) : 정답 데이터와 비교하여 문제점 발견 가능
  * Visual Analytics
    - Dimension Reduction(PCA, LDA, t-SNE, UMAP) + scatter plot (2d, 3d) : text의 경우에는 word2vec 등의 과정을 거쳐야함
    - Data-Data Relation Network Visualization
  * Train/Inference Visualization
    - Loss graph : wandb / tensorboard 등 다양한 툴로 모델의 훈련과정 확인
    - Result : confusion matrix 등

* Image Dataset Visualization
  * 이미지 나열 : [Facet](https://pair-code.github.io/facets/)
  * Patch : bounding box 그릴 때 유용 (`matplotlib.patches`)
    * [Wandb : Image Masks for Semantic Segmentation](https://wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)
    * [How to Do Data Exploration for Image Segmentation and Object Detection](https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection)
    * [COCO dataset explorer](https://github.com/i008/COCO-dataset-explorer)
  * Dimension Reduction + Scatter Plot : t-SNE, UMAP 이용

* **Text Dataset Visualization**
  * [Console output에 Highlight](https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal) : [Termcolor](https://github.com/ikalnytskyi/termcolor) \| [Sty](https://github.com/feluxe/sty) \| [Blessings](https://github.com/erikrose/blessings) \| [Rich](https://github.com/willmcgugan/rich)
  * HTML에 Highlight : `IPython의 HTML` \| [LIT](https://github.com/PAIR-code/lit) \| [Ecco](https://www.eccox.io/) \| [BertViz](https://github.com/jessevig/bertviz) \| [Texthero](https://github.com/jbesomi/texthero) \| [pyLDAvis](https://github.com/bmabey/pyLDAvis) \| [Scattertext](https://github.com/JasonKessler/scattertext) \| [Shifterator](https://github.com/ryanjgallagher/shifterator)

* XAI : 설명 가능한 인공지능
  - [Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers](https://arxiv.org/abs/1801.06889)
  - [XAI using torch](https://captum.ai/)
  - saliency map (heatmap visualization)
  - node-link diagram (network visualization) : [NN-SVG](http://alexlenail.me/NN-SVG/) \| [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) \| [Netscope](http://ethereon.github.io/netscope/quickstart.html)

* AI+Visualization 자료 : [Distill.pub](https://distill.pub/) \| [Poloclub](https://poloclub.github.io/) \| [Google Pair](https://pair.withgoogle.com/) \| [Open AI Blog](https://openai.com/blog/)
* visualization 아이디어를 얻을 수 있는 소스 : [Observable](https://observablehq.com/) \| [Text Visualization Browser](https://textvis.lnu.se/) \| [VisImages](https://visimages.github.io/visimages-explorer/)
* Custom Matplotlib Theme : [Apple Human Interface Guidelines - Color](https://developer.apple.com/design/human-interface-guidelines/ios/visual-design/color/) \| [Google Material Design - Color](https://material.io/design/color/the-color-system.html#color-usage-and-palettes) \| [Color Palettes in Seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html)
