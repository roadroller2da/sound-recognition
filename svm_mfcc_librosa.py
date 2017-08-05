# coding: UTF-8
# (c)r2d.info

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab
import matplotlib.pyplot as plt

import glob
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import resample

def read_mfcc(name_list, base_dir):
    X,y = [],[]
    for label,name in enumerate(name_list):
        for fn in glob.glob(os.path.join(base_dir,name,"*.mfcc.npy")):
            mfcc = np.load(fn)
            num_mfcc = len(mfcc)            
            X.append(np.mean(mfcc[:],axis=1)) # MFCC‚ÌŽžŠÔŽ²•ûŒü‚Ì•½‹Ï‚ð‚Æ‚é
            y.append(label)
    return np.array(X),np.array(y)

def normalisation(cm):
    new_cm = []
    for line in cm:
        sum_val = sum(line)
        new_array = [float(num)/float(sum_val) for num in line]
        new_cm.append(new_array)
    return new_cm

def plot_confusion_matrix(cm,name_list,title):
    pylab.clf()
    pylab.matshow(cm,fignum=False,cmap='Blues',vmin=0,vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(name_list)))
    ax.set_xticklabels(name_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(name_list)))
    ax.set_yticklabels(name_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predict class')
    pylab.ylabel('True class')
    pylab.grid(False)
    filename = "output.png"
    plt.savefig(filename)


#------------------------------------------------------------------------------------

print("****START***********************")

# Labels (Data Directory)
name_list = ["uemura_normal", "tsuchiya_normal", "fujitou_normal"]

# Read Datas
base_dir = "./wav/"
datas,labels = read_mfcc(name_list, base_dir)

# Suffle Datas
# n_samples : Number of samples to generate.
datas,labels = resample(datas, labels, n_samples=len(labels))

print("DataFileNum:" + str(len(datas)))

test_num = 150
svc = LinearSVC(C=1.0)

# TRAINING
svc.fit(datas[test_num:], labels[test_num:])

# TEST
prediction = svc.predict(datas[:test_num])
cm = confusion_matrix(labels[:test_num],prediction)
print(normalisation(cm))
plot_confusion_matrix(normalisation(cm), name_list, "SEIYU")
