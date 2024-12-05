from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request
import os
import cv2
import my_model  # import model dùng để trả kết quả predict

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"          # video khi gửi đến server được lưu ở thư mục static

# yolov6_model = my_yolov6.my_yolov6("weights/fire_detect.pt", "cpu", "data/mydataset.yaml", 640, False)
model, train_transforms = my_model.load_model()


@app.route('/predict', methods=['POST'])
@cross_origin(origins="*")
def predict_video_with_model():
    if 'file' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    # lưu video nhận được từ client và lưu vào local
    video_file = request.files['file']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # nếu chưa có folder lưu video thì tạo 1 folder mới
    video_file.save(video_path)

    # xử lý video
    print(f"Processing video: {video_path}")
    # Cắt video thành face video
    face_video_path = my_model.processing_video(video_path)
    print(face_video_path)

    if not os.path.exists(face_video_path):
        print(f"Skipping {video_path} due to preprocessing error.")
        return jsonify({"error": "Preprocessing error."}), 500

    print("Done preprocessing your video! Please wait a minute to get prediction")

    # Tiền xử lý và dự đoán
    video_dataset = my_model.validation_dataset([face_video_path], sequence_length=40, transform=train_transforms)
    print(len(video_dataset))
    try:
        # for i in range(0, len(video_path)):
        result = my_model.predict(model, video_dataset[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(video_path)
        if os.path.exists(face_video_path):
            os.remove(face_video_path)

    return jsonify(result)


# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
