import cv2
import dlib

def extract_faces_from_video(video_path, output_video_path):
    detector = dlib.get_frontal_face_detector()

    # Đọc video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra video đã mở thành công chưa
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Lấy thông tin về độ phân giải video ban đầu
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Khởi tạo video writer để ghi lại video sau khi cắt khuôn mặt
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video cho .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (112, 112))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển ảnh về ảnh xám (grayscale) để tăng tốc độ phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = detector(gray)

        for face in faces:
            # Lấy tọa độ khuôn mặt
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

            # Cắt khuôn mặt từ frame
            face_image = frame[y:y + h, x:x + w]

            # Resize khuôn mặt về kích thước 112x112
            resized_face = cv2.resize(face_image, (112, 112))

            # Ghi khuôn mặt vào video mới
            out.write(resized_face)

    # Giải phóng bộ nhớ
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video has been processed and save in ", output_video_path)
    # return output_video_path