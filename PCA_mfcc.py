# coding: UTF-8
# (c)r2d.info

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glob
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from sklearn.decomposition import PCA

def read_mfcc(name_list, base_dir):
    X,y = [],[]
    for label,name in enumerate(name_list):
        for fn in glob.glob(os.path.join(base_dir,name,"*.mfcc.npy")):
            mfcc = np.load(fn)
            num_mfcc = len(mfcc)            
            X.append(np.mean(mfcc[:],axis=1)) # MFCCの時間軸方向の平均をとる
            y.append(label)
    # PCAで次元圧縮
    pca = PCA(n_components=3)
    pca.fit(X)
    return np.array(pca.transform(X)),np.array(y)

def normalisation(cm):
    new_cm = []
    for line in cm:
        sum_val = sum(line)
        new_array = [float(num)/float(sum_val) for num in line]
        new_cm.append(new_array)
    return new_cm


#------------------------------------------------------------------------------------

print("****START***********************")

# Labels (Data Directory)
name_list = ["uemura_normal", "tsuchiya_normal", "fujitou_normal"]

# Read Datas
base_dir = "./wav/"
datas,labels = read_mfcc(name_list, base_dir)


# 結果をプロット
fig = plt.figure()
ax = Axes3D(fig)
# 軸ラベルの設定
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
# 表示範囲の設定
#ax.set_xlim(0, 3)
#ax.set_ylim(0, 3)
#ax.set_zlim(0, 3)

colors = ["#ff0000", "#00ff00", "#0000ff"]


for class_num in range(0, len(name_list)):   
    ux=np.array([])
    uy=np.array([])
    uz=np.array([])
    for i in range(0, len(datas)):         # クラスタごとに色を変えてプロットする
        if labels[i] == class_num:
            ux=np.append(ux, datas[i][0])
            uy=np.append(uy, datas[i][1])
            uz=np.append(uz, datas[i][2])
    ax.plot(ux, uy, uz, "o", color=colors[class_num], ms=4, mew=0.5)

plt.show()
filename = "classtering.png"
plt.savefig(filename)