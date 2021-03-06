# dont-be-turtle-ios

MoT Lab에서 진행하는 [거북목 프로젝트](https://github.com/motlabs/dont-be-turtle) 1차 결과물입니다.

![turtle_demo_001](resource/turtle_demo_001.gif)



## 사용 기술

1. Pose Estimation with Machine Learning
2. Moving Average Filter

## 동작 방식

![how it works](resource/how_it_works.png)

## 학습된 모델 준비

[dont-be-turtle](https://github.com/motlabs/dont-be-turtle) 프로젝트에서 학습한 모델을 Core ML 모델로 변환합니다.

#### 모델 포맷

| input shape  | `[1, 192, 192, 3]` |
| ------------ | ------------------ |
| output shape | `[1, 48, 48, 14]`  |

## 참고 & 관련 링크

- [motlabs/dont-be-turtle](motlabs/dont-be-turtle)
- [tucan9389/PoseEstimation-CoreML](https://github.com/tucan9389/PoseEstimation-CoreML)
- [motlabs/iOS-Proejcts-with-ML-Models](https://github.com/motlabs/iOS-Proejcts-with-ML-Models)
- [iOS에서 머신러닝 슬라이드 자료](https://docs.google.com/presentation/d/1wA_PAjllpLLcFPuZcERYbQlPe1Ipb-bzIZinZg3zXkg/edit?usp=sharing)