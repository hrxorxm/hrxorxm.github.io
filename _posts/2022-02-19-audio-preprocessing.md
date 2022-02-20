---
layout: post
title:  "[T아카데미] 디지털신호처리 이해"
subtitle:   "Sampling, STFT, Spectrogram, Mel-Spectrogram, MFCC 등에 대한 강의 내용 정리"
categories: "Speech-Recognition"
tags: []
use_math: true
---

# [T아카데미] 디지털신호처리 이해

## [1강] 디지털신호처리(DSP) 기초 Ⅰ

> 강의 출처 : [[토크ON세미나] 디지털신호처리 이해 1강 - 디지털신호처리(DSP) 기초 I - Sampling, Quantization \| T아카데미](https://youtu.be/RxbkEjV7c0o)

### Audio Task

* 스마트 스피커 덕분에 각광 받고 있다.
* Sound
  * Sound Classification → Activation
  * Auto-tagging (Acoustic Scene / Event Identification)
  * Urban sound tagging system : Audio 와 Spatiotemporal Context 정보를 활용하여 태깅
* Speech
  * Speech Recognition (STT) : 음성 인식 → Input User Interface
    * [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
  * Speech Synthesis (TTS) : 음성 합성 → Output of Interaction
    * [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
  * Speech Style Transfer (STS) : 음성 변환

### 컴퓨터가 소리를 인식하는 방식

* 연속적인 아날로그 신호를 표본화(Sampling), 양자화(Quantizing), 부호화(Encoding)을 거쳐 **이진 디지털 신호(Binary Digital Signal)**로 변환
* 표본화(Sampling) : 초당 샘플링 횟수(Sampling rate)를 정해야 함
  * 1초의 연속적인 시그널을 몇 개의 숫자로 표현할 것인가?
    * 나이키스트-섀년 표본화 : A/D 를 거치고, D/A 로 복원하기 위해서는 표본화 된 신호의 최대 주파수가 두배보다 더 클 때 가능하다.
      * Speech 의 Frequency 분포가 180~2000Hz 이기 때문에, Sampling rate가 4000Hz 이하로 떨어지면 안 된다.
    * 일반적으로 사용되는 주파수 영역대 : 16KHz(Speech), 22.05KHz 44.1KHz(Music)
* 양자화(Quantizing) : Amplitude(진폭)의 real valued 를 기준으로 시그널의 값을 조절하여, 효율적으로 저장
  * Amplitude를 이산적인 구간으로 나누고, signal 데이터의 amplitude를 반올림
    * B bit 의 Quantization : $-2^{B-1}$ ~ $2^{B-1} - 1$
    * ex)Audio CD 의 Quantization (16 bits) : $-2^{15}$ ~ $2^{15} - 1$
    * 위 값들은 보통 -1.0 ~ 1.0 영역으로 scaling 되기도 함
  * **보통 안 하지만, 큰 데이터를 다루거나 경량화 모델이 필요할 때 한다.**
  * Quantization 후 음원에 기계음이 섞이기 시작한다.
* mu-law encoding
  * 사람의 귀는 소리의 amplitude에 대해 log적으로 반응한다.
  * 즉, 작은소리의 차이는 잘잡아내는데 반해 소리가 커질수록 그 차이를 잘 느끼지 못 한다.
  * 이러한 특성을 wave값을 표현하는데 반영해서 작은값에는 높은 분별력(high resolution)을, 큰값끼리는 낮은 분별력(low resolution)을 갖도록 한다.
  * mel-spectrum을 이용할 때는 굳이 이 과정을 하지 않는다.
* Sound Representation
  * Time-domain Representation - Waveform : 시간이 x축, amplitude가 y축
    ![image](https://user-images.githubusercontent.com/35680202/153700679-84038984-6618-4c59-afde-6876d7d2782d.png)
  * Time-Frequency Representation : 시간에 따른 frequency의 변화

### 인간이 소리를 인지하는 방식

* 소리 : 진동으로 인한 공기의 압축
* 파동(Wave) : 진동하며 공간/매질을 전파해나가는 현상 (압축이 얼마나 됐는지)
  * 질량의 이동은 없지만 에너지/운동량의 운반은 존재
* 소리에서 얻을 수 있는 물리량
  * 진폭(Amplitude) : Intensity
  * 주파수(Frequency) : The number of compressed
  * 위상(Phase) : Degress of displacement

  |           물리 음향            |             심리 음향              |
  | :----------------------------: | :--------------------------------: |
  |  Intensity : 소리 진폭의 세기  |        Loudness : 소리 크기        |
  | Frequency : 소리 떨림의 빠르기 | Pitch : 음정, 소리의 높낮이/진동수 |
  | Tone-Color : 소리 파동의 모양  |      Timbre : 음색, 소리 감각      |

  * 주기(period) : 파동이 한번 진동하는데 걸리는 시간 또는 그 길이
  * 주파수(frequency) : 1초 동안의 진동 횟수
    * 소리의 높낮이는 음원의 주파수에 의해 결정
    * 주파수가 높으면 높은 소리, 낮으면 낮은 소리
* 복합파(Complex Wave)
  * 복수의 서로 다른 정현파들의 합으로 이루어진 파형
  * 우리가 사용하는 대부분의 소리들은 복합파
* 정현파(Sinusoidal Wave)
  * 모든 신호는 주파수(frequency)와 크기(magnitude), 위상(phase)이 다른 정현파(sinusolida signal)의 조합으로 나타낼 수 있다.
  * 일종의 복소 주기함수 (복소수(phase)가 있는 주기(frequency)함수)
  * $x(n) \approx \sum_{k=0}^{K} a_k(n) \cos(\phi_k(n)) + e(n)$
    * $a_k$ : instantaneous amplitude
    * $\phi_k$ : instantaneous phase
    * $e(n)$ : residual (noise)

## [2강] 디지털신호처리(DSP) 기초 Ⅱ

> 강의 출처 : [[토크ON세미나] 디지털신호처리 이해 2강 - 디지털신호처리(DSP) 기초 II - STFT, Spectrogram, Mel-Spectrogram \| T아카데미](https://youtu.be/FjYNM3YGFB4)

### 푸리에 변환

* 푸리에 변환(Fourier transform)
  * 조합된 정현파의 합(하모니) 신호에서 그 신호를 구성하는 정현파들을 각각 분리해내는 방법
  * 임의의 입력 신호를 다양한 주파수를 갖는 주기 함수(복수 지수함수)들의 합으로 분해하여 표현하는 것
    ![image](https://user-images.githubusercontent.com/35680202/153701360-d45f0899-f920-4204-b171-48622de562c4.png)
  * $y(t) = \sum_{k= - \inf}^{\inf} A_k \exp(i \cdot 2 \pi \frac{k}{T} t)$ : 푸리에 변환식
    * 입력신호 $y_t$ 가 $\exp(i \cdot 2 \pi \frac{k}{T} t)$ 와 진폭($A_k$)의 선형결합으로 표현된다
    * $k$ : 주기함수들의 개수
    * $A_k = \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} f(t) \exp(-i \cdot 2 \pi \frac{k}{T} t) dt$ : 진폭
* 오일러 공식
  * 지수함수와 주기함수의 관계 : exponential 한 함수를 cos 나 sin 꼴 (주기함수)로 풀어서 쓸 수 있다.
    * $e^{i \theta} = \cos \theta + i \sin \theta$
  * 즉, 푸리에 변환은 입력 signal이 어떤 것이든지 sin, cos 와 같은 주기함수들의 합으로 항상 분해 가능하다.
    * $\exp(i \cdot 2 \pi \frac{k}{T} t) = \cos(2 \pi \frac{k}{T}) + j \sin(2 \pi \frac{k}{T})$
* 푸리에 변환의 결과
  * 실수부와 허수부를 가지는 복소수 반환
  * Spectrum magnitude : 복소수의 절대값 (보통 실수부) → 주파수의 강도
    * $\| X(k) \| = \sqrt{X_{R}^{2}(k) + X_{I}^{2}(k)}$
  * Phase spectrum : 복소수가 가지는 Phase (보통 허수부) → 주파수의 위상
    * $\angle X = \phi(k) = \tan^{-1} \frac{X_I (k)}{X_R (k)}$
* 푸리에 변환의 직교
  * 어떠한 주기함수를 우리는 cos과 sin함수로 표현하게 되었다. 여기서 한가지 재밌는 점은, 이 함수들이 직교하는 함수(orthogonal)라는 점이다.
  * $\{ \exp(i \cdot 2 \pi \frac{k}{T} t) \} = orthogonal$
  * 벡터의 직교는 해당 벡터를 통해 평면의 모든 좌표를 표현할 수 있다. 즉, 위 케이스에서 cos, sin 함수가 사실상 입력신호에 대해서 기저가 되어주는 함수라고 할 수 있다.
* 이산 푸리에 변환(Discrete Fourier Transform, DFT)
  * sampling을 통해 들어온 데이터는 시간의 간격에 따른 소리의 amplitude의 discrete한 데이터
  * 만약 수집한 데이터 $y_n$에서, 이산 시계열 데이터가 주기 $N$으로 반복한다고 할 때, DFT는 주파수와 진폭이 다른 $N$개의 사인 함수의 합으로 표현이 가능하다.
  * $y_n = \frac{1}{N} \sum_{k=0}^{N-1} Y_k \cdot \exp(i \cdot 2 \pi \frac{k}{N} n)$ : input signal
    * $n$ : discrete time index
  * $Y_k = \sum_{n=0}^{N-1} y_n \cdot \exp(-i \cdot 2 \pi \frac{k}{N} n)$ : k번째 frequency에 대한 spectrum의 값
    * $k$ : discrete frequency index
* 고속 푸리에 변환(Fast Fourier Transform, FFT)
  * 위 방식대로 한다면, 시간의 흐름에 따라 신호의 주파수가 변했을 때, 어느 시간대에 주파수가 변하는지 모르게 된다.
* 단시간 푸리에 변환(Short Time Fourier Transform, STFT)
  * FFT의 한계를 극복하기 위해서 STFT는 시간의 길이를 나눠서 푸리에 변환을 한다.
  * $X(l, k) = \sum_{n=0}^{N-1} w(n) x(n+lH) \exp(-2 \pi \frac{k}{N} n)$
    * $N$ : FFT size (window를 얼마나 많은 주파수 밴드로 나누는가)
    * $n$ : window size (window function에 들어가는 sample의 양)
    * $H$ : Hop size (윈도우가 겹치는 사이즈, 일반적으로 $\frac{1}{4}$ 정도 겹침)
    * $w(n)$ : window function
      * 연속성이 깨지는 부분을 보완하기 위해서, 보통 시작/끝이 0인 함수 사용
      * 일반적으로 Hann window가 쓰임
    * Duration : sampling rate를 window로 나눈 값 (신호주기보다 5배 이상 길게 잡아야 한다.)

### 스펙트로그램

* 스펙트로그램(Spectrogram)
  * 어떤 frequency 영역대가 강한지(활성화 되어있는지) 시각화
  * wave form : 시간 축에 따른 amplitude
  * spectrogram : 시간 축에 따른 frequency + amplitude (+ phase)
* 인간의 청각경로(Auditory Pathway)
  * High-frequency sound에 맵핑된 것 : base
  * Low-frequency sound에 맵핑된 것 : apex
  * 보통 인간은 저주파를 더 잘 인식한다.
* 멜 스펙트로그램(Mel-Spectrogram)
  * 사람은 인접한 주파수를 크게 구별하지 못한다. (categorical한 구분을 하기 때문)
  * 멜 스펙트럼은 주파수 단위를 멜 단위로 바꾼 것
  * 멜 필터(Mel-Filter bank)는 저주파 주변에서 얼마만큼 에너지가 있는지를 알려줌
  * mel-scaled bin을 FFT size 보다 조금 더 작게 만드는 것이 일반적

## [3강] 디지털신호처리(DSP) 기초 Ⅲ

> 강의 출처 : [[토크ON세미나] 디지털신호처리 이해 3강 - 디지털신호처리(DSP) 기초 III - MFCC, Auditory Filterbank \| T아카데미](https://youtu.be/kiTHOCmWPsg)

### MFCC

* MFCC(Mel-Frequency Cepstral Coefficient)
  * Filter Bank는 모두 Overlapping 되어 있기 때문에 Filter Bank 에너지들 사이에 상관관계가 존재
  * Mel-Spectrum 에 log 적용
  * Mel-log-Spectrum list 전체에 DCT(Discrete Cosine Transform) 적용
    * DCT(Discrete Cosine Transform) : n개의 데이터를 n개의 코사인 함수의 합으로 표현하여 데이터의 양을 줄이는 방식
    * 저주파수에 에너지가 집중되고 고주파수 영역에 에너지가 감소
  * 얻어진 Coefficients 에서 앞에서부터 N개만 남기고 버린다.
    * 여기서 26개 DCT Coefficient 들 중 12만 남겨야 하는데, 그 이유는 DCT Coefficient 가 많으면, Filter Bank 에너지의 빠른 변화를 나타내게 되고, 이것은 음성인식의 성능을 낮추게 된다.
  * 음색을 잡거나 고유한 화자를 분류할 때 주로 쓰인다.

## [4강] 디지털신호처리(DSP) 실습 Ⅰ

> 강의 출처 : [[토크ON세미나] 디지털신호처리 이해 4강 - 디지털신호처리(DSP) 실습 I - Data augmentation \| T아카데미](https://youtu.be/VzR0hBVZvRA)

### Data Augmentation

* 증강 방법 3가지
  * wave form 에서 노이즈를 섞거나 pitch 등을 바꾸기
    * `librosa.effects.pitch_shift`
  * spectrogram 에서 마스킹하기 (speech recognition에서 좋음)
    * `torchaudio.transforms.FrequencyMasking`
    * `torchaudio.transforms.TimeMasking`
  * data split 하기 (classification에서 좋음) (예: 30초짜리를 5초씩 끊어서 보기)
  * 추가) mixup 등도 적용 가능할 것

## [5강] 디지털신호처리(DSP) 실습 Ⅱ

> 강의 출처 : [[토크ON세미나] 디지털신호처리 이해 5강 - 디지털신호처리(DSP) 실습 II - DataLoader \| T아카데미](https://youtu.be/oFdjv1CJRWo)

### DataLoader

* 데이터 로드 방법 2가지
  * 배치마다 멜 스펙트로그램으로 변환 : 저장 공간이 넉넉하지 않거나, 멜 스펙트로그램을 조금씩 바꿔서 만들어보면서 실험할 때 좋음
  * 멜 스펙트로그램으로 저장해놓고 로드 : 저장 공간이 넉넉하고, 멜 스펙트로그램은 변경하지 않고 실험할 때 좋음
