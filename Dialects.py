import librosa
import os
import numpy

datasets = []
filepath = "/Database/"
filename= os.listdir(filepath)
for file in filename:
    wav = filepath+file
    y, sr = librosa.load(wav, sr=16000)
    mfcc = librosa.feature.mfcc(y, sr=16000)
    vector = mfcc.mean(1)#平均之后的特征向量
    datasets.append(vector)

#print(datasets)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = datasets#二维
y = ['普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话', '普通话','北京', '北京', '北京', '北京', '北京', '北京', '北京', '北京','东北', '东北', '东北', '东北', '东北', '东北', '东北', '东北','粤语', '粤语', '粤语', '粤语', '粤语', '粤语', '粤语', '粤语','江浙', '江浙', '江浙', '江浙', '江浙', '江浙', '江浙', '山东', '山东', '山东', '山东', '山东', '山东', '山东','云贵川', '云贵川', '云贵川', '云贵川', '云贵川', '云贵川', '云贵川','天津', '天津', '天津', '天津', '天津', '天津', '天津', '天津','西北','西北','西北','西北', '西北','西北', '西北']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
result = knn.predict(X_test)

count = 0
for item1,item2 in zip (result, y_test):
    if item1 == item2:
        count += 1
print ("Accuracy: " + str(float(count)/float(len(y_test))))

import matplotlib.pyplot as plt
