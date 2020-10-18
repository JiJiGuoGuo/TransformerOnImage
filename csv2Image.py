#这个文件用来将fer2013解析成图片，并按照3个用途进行分类文件夹，每个文件夹有7种不同的表情
import os
import pandas as pd
import numpy as np
from PIL import Image#图像进行处理的基础功能
emotions = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'nomral'}

#七种表情的分类
dataset_path=r'.\Datasets\archive'
csv_file = os.path.join(dataset_path,'fer2013.csv')


fer2013 = pd.read_csv(csv_file)
for i in range(len(fer2013)):
    emotion_data = fer2013.loc[i][0]
    image_data = fer2013.loc[i][1]
    usage_data = fer2013.loc[i][2]
    image = np.asarray([float(p) for p in image_data.split()]).reshape(48,48)

    #图片要保存的文件夹
    usagePath = os.path.join(dataset_path,usage_data)
    if not os.path.exists(usagePath):
        os.mkdir(usagePath)
    imagePath = os.path.join(usagePath, emotions[str(emotion_data)])
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)


    imageName = os.path.join(imagePath,str(i)+'.jpg')
    Image.fromarray(image).convert('L').save(imageName)

