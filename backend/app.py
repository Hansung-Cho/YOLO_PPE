from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = FastAPI()

# CORS: 아무 origin에서나 호출 허용 (로컬 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 나중에 특정 도메인으로 줄이면 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작할 때 한 번만 모델 로드
model = YOLO("best.pt")  # best.pt 경로 맞게 수정

@app.get("/")
def root():
    return {"message": "PPE detection API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 업로드된 이미지 파일 읽기 (bytes)
    image_bytes = await file.read()

    # 2. bytes -> numpy 배열(BGR 이미지)로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. YOLO 추론
    results = model.predict(source=img, conf=0.5, verbose=False)
    res = results[0]

    detections = []
    danger_flag = False
    names = model.names  # 클래스 이름 딕셔너리

    if res.boxes is not None:
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()             # [x1, y1, x2, y2]
        confs = boxes.conf.cpu().numpy()            # confidence
        clss = boxes.cls.cpu().numpy().astype(int)  # class index

        for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, clss):
            detections.append({
                "class_id": int(cls_idx),
                "class_name": names[int(cls_idx)],
                "confidence": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
            })

    # 4. 박스가 그려진 이미지 생성
    annotated = res.plot()  # BGR numpy array

    # 5. annotated 이미지를 JPG로 인코딩 후 base64로 변환
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "num_detections": len(detections),
        "detections": detections,
        "image_base64": img_base64,
    })
