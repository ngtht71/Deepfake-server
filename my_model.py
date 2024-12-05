# import libraries
# !pip3 install face_recognition

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition

# Model with feature visualization
from torch import nn
from torchvision import models

# import processing.py
import preprocessing

# Load model and Code for making prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


# define model
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("gpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image * 255)
    return image


# hàm tiền xử lý dữ liệu video -> cắt video tập trung vào face
def processing_video(video_path):
    face_video_path = video_path.replace(".mp4", "_face.mp4")
    preprocessing.extract_faces_from_video(video_path, face_video_path)
    return face_video_path


# class xử lý dữ liệu video đầu vào
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=40, transform=train_transforms):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        # first_frame = np.random.randint(0,a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            # Kiểm tra nếu frame không phải RGB
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


# hàm dự đoán đầu vào với model đã xây
def predict(model, img):
    fmap, logits = model(img.to(device))
    sm = nn.Softmax(dim=1)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    result = {
        "prediction": "REAL" if prediction.item() == 1 else "FAKE",
        "confidence": confidence
    }
    return result


def load_model():
    model = Model(2).to(device)
    model_path = 'data/Models/model_89_acc_40_frames_final_data.pt'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, train_transforms


# test predict
# path_to_videos = ['data/examples_DF/id61_0006.mp4',
#                   'data/examples_DF/id61_0007.mp4',
#                   'data/examples_DF/id61_0008.mp4',
#                   'data/examples_DF/id61_0009.mp4',
#                   # 'test_video/face_video.mp4'
#                   ]
#
#
# model, train_transforms = load_model()
#
# # example use model
# video_dataset = validation_dataset(path_to_videos, sequence_length=40, transform=train_transforms)
# for i in range(0, len(path_to_videos)):
#     print(path_to_videos[i])
#     prediction = predict(model, video_dataset[i])
#     print(prediction)
