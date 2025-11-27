from ultralytics import YOLO
import cv2

# 1. 모델 로드 (Colab에서 가져온 best.pt 경로)
model = YOLO("best.pt")  # 같은 폴더에 있으면 이렇게만 써도 됨

# 2. 웹캠 열기 (0 = 기본 카메라)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    # 3. YOLO로 추론
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # 4. bounding box가 그려진 이미지 얻기
    annotated_frame = results[0].plot()  # numpy array (BGR)

    # 5. 화면에 출력
    cv2.imshow("PPE Detection", annotated_frame)

    # 6. 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. 자원 해제
cap.release()
cv2.destroyAllWindows()
