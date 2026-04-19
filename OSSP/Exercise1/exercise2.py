import cv2

# Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 웹캠 실행
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 얼굴 박스 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # 화면 출력
    cv2.imshow('Face Detection (Webcam)', frame)

    # 종료 키 (q 또는 ESC)
    key = cv2.waitKey(1)
    if key == 27 or (key & 0xFF == ord('q')):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()