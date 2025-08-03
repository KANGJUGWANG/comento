# 🚦 교통량 예측 모델 개발 및 로컬 인터페이스 구현

본 프로젝트는 **Comento의 '데이터사이언티스트와 함께하는 인공지능 프로젝트 A to Z: 모델링부터 서빙까지' 커리큘럼**을 기반으로 진행되었습니다.  
진행 과정에서 **프로젝트 주제, 주차별 목표, 코드 예시 및 피드백이 일부 제공**되었으며,  
이를 바탕으로 **데이터 처리, 모델 구성 및 학습, 시각화 구현, 발표자료 작성 등은 개인이 주도적으로 설계 및 구현**하였습니다.

---

## 📌 프로젝트 개요

- **주제:** CCTV 이미지 기반 도로 교통량 예측
- **기간:** 2025년 6월 13일 ~ 2025년 7월 11일
- **주요 기술:** Python, Jupyter Notebook, YOLOv8, FastAPI, Streamlit, Anaconda

---

## 📁 폴더 구조

```
/src/                       # 실행 소스코드 (데이터 분할, 모델 훈련, API, 인터페이스)
/docs/                      # 발표자료 및 성능 비교 자료
/origin_synthetic_data_set/ # 원본 합성 데이터
/origin_actual_data_set/    # 원본 실제 데이터

# 데이터 분할 실행 시 생성
/synthetic_data_set/        # 정리된 합성 데이터
/actual_data_set/           # 정리된 실제 데이터

# 모델 학습 실행 시 생성
/src/runs/detect/
    └── train_1st_yyyymmdd_HHMM  # 1차 학습 결과 (합성 데이터)
    └── train_2nd_yyyymmdd_HHMM  # 2차 학습 결과 (실제 데이터)
    └── trainX                  # 중단된 학습 결과
```

---

## ⚙️ 실행 환경

- Python: `3.11.13`
- CUDA: `Not available`
- cuDNN: `90701`
- PyTorch: `2.7.1+cu128`
- Ultralytics YOLO: `8.3.158`
- Streamlit: `1.46.1`
- FastAPI: `0.115.14`
- GPU: `NVIDIA GeForce RTX 3070 Laptop (8GB)`
- OS: `Windows 10 / Windows-10-10.0.26100-SP0`

---

## 🚀 실행 방법

### 1. 데이터 다운로드

- **합성 데이터:** [DACON - CCTV 이미지 데이터](https://dacon.io/competitions/official/236107/overview/description)  
  → `/origin_synthetic_data_set/` 하위에 저장

- **실제 데이터:** [Kaggle - UA-DETRAC](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset)  
  → `/origin_actual_data_set/content/UA-DETRAC/DETRAC_Upload/` 에 저장

---

### 2. 경로 설정

- `/src/config.py` 파일에서 사용자의 로컬 경로에 맞게 수정
- 이후 모든 소스에서 해당 경로가 참조됨 (경로 통일)

---

### 3. 데이터 정리

- `src/Data_Segmentation.ipynb` 실행  
  → YOLO 학습 포맷으로 자동 정리된 `synthetic_data_set`, `actual_data_set` 생성됨

---

### 4. 모델 학습

- `src/yolo11s_train.py` 실행  
  → `1차/2차 학습`, `이어 학습` 여부 선택 후 진행  
  → 학습 완료 시 `/src/runs/detect/train_...` 폴더 자동 생성

---

### 5. 모델 테스트 및 인터페이스 실행

- **로컬 API 실행:**  
  `src/background_api.py` 실행  
  → FastAPI 기반 모델 예측 테스트 가능

- **웹 인터페이스 실행:**  
  `src/streamlit.py` 실행  
  (단, `background_api.py`가 실행 중이어야 동작)

- **테스트 이미지 저장 (옵션):**  
  `background_api.py`의 `line-30`  
  → `save=False` → `True` 변경 시 바운딩 이미지 저장됨

- **신뢰도(confidence) 조정 (옵션):**  
  `line-30`의 `conf=0.7` 값을 변경 가능

---

## ✅ 라이선스 및 출처

- 본 프로젝트는 Comento 커리큘럼을 기반으로 하였으며, 개인적인 학습 및 포트폴리오 목적으로 제작되었습니다.

