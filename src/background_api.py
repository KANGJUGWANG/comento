##### jupyter notebook code를 .py형태로 저장#####
## 패키지 불러오기 
import os
import uvicorn
import nest_asyncio
from PIL import Image
from io import BytesIO
from pyngrok import ngrok
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form

from ultralytics import YOLO ## yolo모델용
import base64 ## Base64 인코딩용
import cv2 ## OpenCV 이미지 처리용
from config import get_yolo_model_path  

## api생성 및 모델 로드
# fastAPI 생성
app = FastAPI()
# 모델 로드
yolo11s_model = YOLO(get_yolo_model_path())
async def yolo_model(image:Image.Image):
    """
    * 이미지를 입력받아 분석할 모델
    input 
    image(Image.Image) : 처리할 이미지
    output
    detection(ultralytics.engine.results.Results) : 단일 이미지의 정보 객체
    """
    return yolo11s_model.predict(image, conf=0.7,save=False)[0]
    
## cctv 분석 결과를 도출하는 api함수 
@app.post('/analysis')
async def cctv_analysis_api(
        file: UploadFile = File(...),## 필수로 이미지 파일을 전달 받아야함
        max_car:int = Form(10), ## 점유율의 기준이 될 최대 차량수
        streamlit_json:bool = Form(False) ## streamlit에서 접속여부(반환 형식 차이)
):
    """
    * cctv이미지를 통해 도로의 차량 정체를 분석하는 api
    input
    file(uploadfile) : 분석할 cctv이미지(필수)
    output
     : 이미지에서 차량의 숫자
    """
    try:
        # 비동기적으로 파일 읽기
        image_bytes = await file.read()
        # 2. 바이트 데이터를 PIL Image 객체로 변환 (모델 입력 형식에 맞춤)
        image_pil = Image.open(BytesIO(image_bytes))
        # 이미지 단일 객체 반환
        detect = await yolo_model(image_pil)
        # --- 탐지 결과 데이터 추출 및 가공 (`detect` 객체 처리) ---
        processed_detections = []
        if detect and detect.boxes:
            for box_data in detect.boxes:
                class_id = int(box_data.cls[0].item())
                class_name = detect.names[class_id]
                confidence = float(box_data.conf[0].item())
                bbox_coords = box_data.xyxy[0].tolist() # [x_min, y_min, x_max, y_max]

                processed_detections.append(
                    {
                        "class_name":class_name,
                        "confidence":confidence,
                        "bbox":bbox_coords,
                        "class_id":class_id
                    })
        car_count = len(detect.boxes)
        ## 차량 점유울 판단 기준 
        # 차량수 : n
        # 점유율 = (n)/(max)
        # 판단 기준 : 원활 <= 0.3 , 0.3<서행 <= 0.7, 0.7 < 정체
        share = round(car_count/max_car,3)

        encoded_image = None
        buffered = BytesIO()
        if detect: # 결과 객체가 있다면
            # 찾은 객체를 바운딩한 np이 데이터 
            image_np = detect.plot() 
            ## 모델 반환 이미지는 BGR형식이므로 RGB로 변경
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # 이미지를 pil형식 변환 
            # 임시버퍼로 이미지 저장(전송 가능한 바이트로 저장&형식 지정) 후 Base64 인코딩
            Image.fromarray(image_np).save(buffered, format="PNG")            
        else:# 결과가 데이터가 없으면 전달 받은 이미지 그대로 출력
            image_pil.save(buffered, format='PNG') 
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        ## api 반환 
        if share<=0.3:
                analysis = f"원활({share})" 
        elif share<=0.7:
            analysis =f"서행({share})"
        else:
            analysis = f"정체({share})"
        if streamlit_json==True:## streamlit용 json반환 
            return {
                "filename":file.filename,
                "yolo_image" : encoded_image,
                "detections":processed_detections, # 탐지된 객체 상세 정보
                "car_count":car_count, # 차량 수
                "share":share, # 점유율
                "analysis":analysis, # 혼잡도 분석
                }
        else:## FastAPI용 string 반환
            return analysis
        
    except Exception as e:
        print(f"error :{e}")
        raise HTTPException(status_code=500, detail="...")


nest_asyncio.apply() 
port = 8000
print(f"FastAPI 서버 시작 중... http://localhost:{port}/docs (API 문서)")
uvicorn.run(app, host="0.0.0.0", port=port)