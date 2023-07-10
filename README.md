# AiffelDLThon
Aiffel 온라인 4기/NLP/딥러닝경기 팀의 결과물 저장소 입니다.

## 팀원 소개
팀장 : 김동규
팀원 : 남희정
팀원 : 소용현
팀원 : 백민홍
팀원 : 신성윤

## 문제 소개
DKTC (Dataset of Korean Threatening Conversations) 데이터셋을 이용한 텍스트 분류

데이터셋: [DKTC dataset](https://drive.google.com/drive/folders/1UIkI5ipERMAEKfjm2N-vdUbxdj1vZoFz?usp=sharing)

협박, 갈취, 직장 내 괴롭힘, 기타 괴롭힘 4가지 대화 유형 Class를 분류하는 딥러닝 모델을 만는 것이 목표

## 결과물 소개
최종 결과물: TorchBERT.ipynb
실행시 그래픽카드 사양을 고려하시고, optuna 관련 코드는 **반드시 제외**하고 실행하세요.

발표 자료: 딥러닝경기_0710_v0.4.pptx

저희는 팀으로서 EDA 과정을 나누어서 수행했습니다.
최종 결과물에 포함되지 않은 여러 EDA 기법들은 아래에 설명된 노트북들에 나눠져 있습니다.

기본 학습 코드 자체가 굉장히 길고 복잡하기 때문에 불가피하게 파일을 나누기로 결정했으니 이점 양해부탁드립니다.

## 모델 연구 자료
AutoKerasPractice: 대표적인 AutoML 프레임워크 중 하나인 AutoKeras를 이용하여 모델을 구현하고 실험한 노트북입니다.

BaseDNNModels: 가장 기본적인 딥러닝 모델인 MLP, CNN, RNN 기반 모델을 실험하는 노트북입니다.

base: 데이터를 가장 빠르게 분석하기 위해 나이브 베이즈 모델로 실험한 노트북입니다.

## 전처리 연구 자료
preprocess_lab-1700_syh: 모든 전처리 기능과 임베딩 그리고 머신러닝과 딥러닝의 차이를 비교 할 수 있는 테스트 베드 입니다. 약간의 수정만으로 이를 분석 할 수 있게 설계되었으며, 이 버전에서는 모든 것을 자동으로 테스트하는 기능이 포함되었습니다. 발표자료의 표도 이 노트북을 이용한 결과물입니다.

preprocess_lab-1400_syh: 초기 버전에 있던 임베딩 기능의 오류가 해결 되었고, stopward에 대한 전처리도 포함되었습니다.

preprocess_lab: 전처리 테스트 베드의 가장 초기 버전 노트북입니다. 팀원들이 전처리별 성능 변화에 대한 검증을 쉽게 할 수 있게 설계하였지만, 여러 오류가 있어서 결국 팀원들의 도움을 받아 수정하였습니다.

nb_stopword: 데이터셋 내의 불용어 분석 및 경향성 분석이 이루어진 노트북입니다.

lengDist: 데이터셋에 존재하는 단어의 길이를 분석한 노트북입니다.

## 임베딩 연구 자료
practice_embedding-fasttext_output_final: fasttext레이어 대한 실험과 동시에 테스트셋을 적용하는 작업이 이루어진 노트북입니다.

practice_embedding-fasttext: fasttext레이어에 대한 실험이 진행된 노트북입니다.

practice_embedding: word2vec에 대한 실험이 진행된 노트북입니다.

## 데이터
train.csv: 학습용 데이터 입니다.

output.csv: 테스트 데이터 입니다. 사람에 의해 라벨링된 데이터입니다.