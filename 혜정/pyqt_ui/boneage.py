print('Importing modules about 10 seconds')

from tqdm import tqdm
import bone 
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import math
from datetime import datetime
import cv2
import re
import torch
import os
import tensorflow.keras as tf
import warnings
warnings.filterwarnings('ignore')
# path ----------------------------------------------------------------------
model_path = './weight/model.pt'
tjnet_path = './weight/tjnet24.h5'

for i in tqdm(range(1),desc='Model Loading...'):
    yolo = torch.load(model_path, map_location='cpu')
    tjnet = tf.models.load_model(tjnet_path, compile=False)

while True:
    try:
        path = input("Drag a photo here. (Shutdown : exit ) > ")
        if path == 'exit':
            sys.exit()

        while True:    
            gender_text = input("Choose your gender. Female / Male > ")
            if gender_text == 'Female':
                gender = 0
                break
            elif gender_text == 'Male':
                gender = 1
                break
            else:
                print('Invalid value. Please try again.')
                continue
                
        now = datetime.now()
        formattedDate = now.strftime("%Y%m%d_%H%M%S")
        filename = formattedDate +'.jpg'
        save_path = './img_save/' + filename

        for i in tqdm(range(1) , desc="Image Preprocessing... "):
            original_img = bone.read_img(path)
            mask = bone.make_mask(original_img)
            masked = bone.cut_mask(original_img, mask)
            rotated_img = bone.img_rotation(masked)
            bone_img = bone.Decomposing(rotated_img,60,55,50,25)

        cv2.imwrite(save_path, bone_img)
        crops, img, result = bone.yolo_crop_img(save_path, yolo) 
        X = bone.out_crop_img(crops, gender) 
        for i in tqdm(range(1), desc='Calculating... '):
            prediction_BA = bone.predict_zscore(X , tjnet)
            prediction_BA = prediction_BA.round(2) 
            print('Predicted Bone Age : ' , prediction_BA)

        # ------------------------------------------------------------
        # 이미지 예측값 넣고 띄우기
        predict_result = bone.bone_age_window(img, gender_text, prediction_BA)
        cv2.imshow('bone_img', predict_result)
        cv2.waitKey()

        while True:
            stop = input("Shutdown [y/n]. > ")
            if stop == 'y':
                # print('다시 시작하시려면 커널 재시작 하세요')
                sys.exit()
            elif stop == 'n':
                print('Restarting..')
                break
            else:
                print('Invalid value. Please try again')
                continue
                
    except Exception as e : 
        print('error', e)
        print('fixed error and try again')