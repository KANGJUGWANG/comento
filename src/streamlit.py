import streamlit as st # type: ignore # Streamlit 라이브러리
import requests         # FastAPI API에 HTTP 요청을 보내기 위해
import io               # BytesIO를 위해
from PIL import Image   # 이미지 처리 (Streamlit의 st.image 위젯에 필요)
import json           # (requests.response.json()이 알아서 파싱하므로 직접 필요하지는 않음)
import base64 ## Base64 디코딩용
# --- 1. FastAPI 서버 정보 설정 ---
FASTAPI_ENDPOINT_URL = "http://localhost:8000/analysis" # FastAPI 서버의 /analysis 엔드포인트 주소

# --- 2. Streamlit 앱 UI 구성 ---
st.set_page_config(
    page_title="차량 혼잡도(도로 점유율) 분석 대시보드", # 웹 페이지 제목
    layout="centered", # 페이지 레이아웃 
)

st.sidebar.title("차량 혼잡도 분석") # 앱 제목
st.sidebar.markdown("cctv 이미지를 분석하여 차량 탐지 및 정체도를 판단합니다.") # 마크다운 텍스트
st.sidebar.number_input("점유율 기준 최대 차량수", min_value=0, max_value=100, value=10, step=1)

# --- 3. 이미지 파일 업로드 위젯 ---
uploaded_file = st.sidebar.file_uploader("여기에 CCTV 이미지 파일(PNG)을 업로드해주세요...", type=["png"])

# --- 4. 파일이 업로드되었을 때의 로직 ---
if uploaded_file is not None:
    # 업로드된 이미지를 대시보드에 미리보기
    st.image(uploaded_file, caption="업로드된 이미지", use_container_width =True)
    st.write("이미지를 분석 중입니다. 잠시만 기다려 주세요...")

    # 파일 내용을 바이트(bytes)로 읽기
    image_bytes = uploaded_file.getvalue()

    # FastAPI로 HTTP POST 요청 전송
    # ('파일 필드 이름', (파일명, 파일 내용(바이트), Content-Type))
    files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)} # 'max_car'는 FastAPI에서 처리할 수 있는 최대 차량 수를 지정하는 파라미터입니다.
    add_data = {'max_car':"10",'streamlit_json':"True"}
    try:
        # requests 라이브러리를 사용하여 FastAPI API에 요청을 보냅니다.
        response = requests.post(FASTAPI_ENDPOINT_URL, files=files, data=add_data)
        
        # HTTP 응답 코드가 200번대가 아니면 예외를 발생시킵니다 (4xx, 5xx 에러).
        response.raise_for_status() 
        
        analysis_results = response.json() # JSON 응답을 파이썬 딕셔너리로 파싱
        
        st.success("분석 완료!")
        
        # --- 5. 분석 결과 표시 ---
        st.subheader("분석 결과:")
        # 특정 결과만 추출하여 사용자 친화적으로 표시
        st.write(f"**파일명:** {analysis_results.get('filename', 'N/A')}")
        yolo_image_base64 = analysis_results.get('yolo_image')
        if yolo_image_base64:
            img_bytes_decoded = base64.b64decode(yolo_image_base64)
            pil_image_decoded = Image.open(io.BytesIO(img_bytes_decoded))
            st.image(pil_image_decoded, caption="분석한 이미지", use_container_width=True) #
        st.write(f"**탐지된 차량 수:** {analysis_results.get('car_count', 'N/A')}")
        st.write(f"**탐지된 점유율:** {analysis_results.get('share', 'N/A')}")
        st.write(f"**탐지 결과:** {analysis_results.get('analysis', 'N/A')}")

        detections = analysis_results.get('detections', []) # 탐지된 객체 목록 가져오기
        if detections:
            st.subheader("탐지된 차량 목록:")
            # Pandas DataFrame으로 변환하여 보기 좋게 표시합니다.
            import pandas as pd 
            df_detections = pd.DataFrame(detections)
            st.dataframe(df_detections)
        else:
            st.write("이미지에서 차량이 탐지되지 않았습니다.")
        # col2.json(analysis_results) #디버깅용 전체 JSON
        
    except Exception as e:
        st.error(f"분석 중 예기치 않은 오류 발생: {e}")