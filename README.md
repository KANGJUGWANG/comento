# comento
# 5주간 AI프로젝트 : 교통량 예측 모델 개발 및 로컬 인터페이스 구현
프로젝트 목표 : cctv 이미지 데이터 기반으로 도로 교통량 예측
기간 : 2025년 6월 13일  ~ 2025년 7월 11일
주요 기술 : python, jupyter notebook, anaconda3, yolo11s, Fast API, streamlit

# 폴더 구조
/src/ : 실행 소스코드(모델훈련, 데이터 분할, 로컬 api, 사용자 인터페이스)
/docs/ : 발표자료, 모델 성능 비교자료
/origin_synthetic_data_set/ : 데이터 분할전 합성 데이터 폴더
/origin_actual_data_set/ : 데이터 분할전 실제 데이터 폴더

-데이터 분할 실행시 생성-
/synthetic_data_set/ : 모델 학습에 적용할 정리된 합성 데이터 폴더 
/actual_data_set/ : 모델 학습에 적용할 정리된 실제 데이터 폴더

-모델 학습 실행시 생성-
/src/runs/detect/... : train_1st_yyyymmdd_HHMM(합성 데이터 학습 결과), train_2nd_yyyymmdd_HHMM(실제 데이터 학습 결과), trainX(모델 학습 강제중단)


# 실행 방법

사전 준비 : 
- 데이터 준비
** 합성 데이터 다운로드
** https://dacon.io/competitions/official/236107/overview/description
** 실제 데이터 다운로드
** https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset

* \origin_actual_data_set\content\UA-DETRAC\DETRAC_Upload\...(images, labels) 경로로 저장
* \origin_synthetic_data_set\...(test, train, classes.txt, data_set.yaml,sample_submission.csv) 경로로 저장

- 폴더 주소 수정(사용자 선택)
*config.py을 수정하여 학습 데이터 및 프로젝트 관련 실행코드에 적용되는 경로 수정 가능

데이터 자동 정리 : Data_Segmentation.ipynb 실행시 synthetic_data_set,actual_data_set 생성 및 yolo모델이 학습 가능한 형태의 파일로 구성

모델 학습 : yolo11s_train.py 실행시 1차학습, 2차학습, 일괄학습 선택지 출력 -> 이어서 학습 선택지 출력(기존 학습이 중단(ctrl+c)되었다면 이어서 학습 가능) 
- 학습 완료 이후 train_학습 단계_완료 날짜 폴더 생성

학습 모델 검증 : background_api.py 실행시 로컬 api로 학습된 모델 테스트 가능 
- streamlit.py파일 실생시 로컬 인터페이스로 모델 테스트 가능(background_api.py실행 중에만 가능)
- background_api.py 파일의 line-30에서 save=False를 True로 변경하면 객체가 바운딩된 테스트 이미지 생성
- background_api.py 파일의 line-30에서 conf=0.7를 수정하여 객체의 신뢰도를 수정 가능

