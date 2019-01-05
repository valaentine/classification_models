#https://github.com/qubvel/classification_models
#!pip install opencv-python
#!apt update && apt install -y libsm6 libxext6
#!apt-get install -y libxrender-dev
#!pip install keras
#!pip install scikit-image

import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import decode_predictions
from classification_models import ResNet152
from classification_models import ResNet50
from classification_models import ResNet34
from classification_models.resnet import preprocess_input
import os
from tensorflow.python.client import device_lib
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# print("---wait---\n")
# print(device_lib.list_local_devices())

# 이미지 불러오기
x = cv2.imread('./imgs/tests/4.jpg')
x_image = cv2.resize(x, (224, 224))
x_img_rgb = cv2.cvtColor(x_image, cv2.COLOR_BGR2RGB)
x = preprocess_input(x_image)
x = np.expand_dims(x, 0)

# resnet pre-trained weight 불러오기
# load model - This will take < 10 min since we have to download weights (about 240 Mb)
# model = ResNet152(input_shape=(224,224,3), weights='imagenet', classes=1000)
model = ResNet50(input_shape=(224,224,3), weights='imagenet', classes=1000)
# model = ResNet34(input_shape=(224,224,3), weights='imagenet', classes=1000)

# processing image
y = model.predict(x)

# 결과
predictions_array = decode_predictions(y)[0]

# 시각화
plt.imshow(x_img_rgb)

for pred in predictions_array:
    _,class_name, pred_num = pred
    text = class_name + ': ' + str(pred_num)
    print(text)

plt.imshow(x_img_rgb)  
