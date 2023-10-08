import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import streamlit as st
import tempfile


class NetRnn(nn.Module):

    def __init__(self):
        super(NetRnn, self).__init__()

        conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2)
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.feat_extractor=nn.Sequential(conv1,nn.ReLU(),pool1,conv2,nn.ReLU(),pool2)
        self.rnn = nn.LSTM(input_size=64*7*7, hidden_size=128, num_layers=2,batch_first=True)
        self.fc1 = nn.Linear(128, 5)

    def forward(self, x):
        batch_size, frames, channels, h, w = x.shape
        x = x.view(batch_size*frames,channels,h,w)
        y = self.feat_extractor(x)
        y = self.bn(y)
        y = y.view(batch_size,frames,-1)
        emb, (hn, cn) = self.rnn(y)
        out = self.fc1(emb[:,-1,:])
        return out

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    # Covvert to float32
    transforms.Lambda(lambda x: x.float())
])

model = NetRnn()
model.load_state_dict(torch.load('rnn_model.pt', map_location='cpu'))
model.eval()

def preprocess(videopath):
    vidcap = cv2.VideoCapture(videopath)

    st.write('Frames from uploaded video:')
    
    success,image = vidcap.read()
    image_list = []
    count = 0
    while success:
        image = transform(image)
        image_list.append(image)
        success,image = vidcap.read()
        count += 1
    # choose 16 equidistant frames
    image_list = np.array(image_list)
    image_list = image_list[np.linspace(0, count-1, 16, dtype=int)]
    images = np.array(image_list)
    return images

def predict(videopath):
    images = preprocess(videopath)
    images = torch.from_numpy(images)
    images = images.unsqueeze(0)
    with torch.no_grad():
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
    
predictions_map = {
    0: "laugh",
    1: "pullup",
    2: "pick",
    3: "punch",
    4: "pour"
    }


st.title("Video Action Classification")
st.write("A minimal classifier for video activity using PyTorch and Streamlit can classify between 5 classes of actions - [ laugh , pick , pour , pullup , punch ].")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write('Uploaded video:')
    st.video(tfile.name)

    with st.spinner("Fetching Results..."):
        predictions = predict(tfile.name)
        print(type(predictions))
        st.write(predictions_map[predictions])

