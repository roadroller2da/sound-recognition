# coding: UTF-8
# (c)r2d.info

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy
from scipy import io
from scipy.io import wavfile
import glob
import numpy as np
import os

import librosa.display
import librosa



def write_mfcc(mfcc,fn):
    base_fn,ext = os.path.splitext(fn)
    data_fn = base_fn + ".mfcc"
    np.save(data_fn,mfcc)


# ceps : Mel-cepstrum coefficients
# mspec: Log-spectrum in the mel-domain.
# spec : spectrum magnitude

def create_ceps(fn):
    sample_rate,X = io.wavfile.read(fn)
    print(fn + ":" + str(sample_rate) + "Hz")
    fs=40
    mfcc_feature = librosa.feature.mfcc(X,n_mfcc=fs)
    # (éüå≥êî, BINêî)
    # BINêî = SampleRate*Lengh of Audio / SlideSize(Default512)
    print(mfcc_feature.shape)
    # âÊëúê∂ê¨
    librosa.display.specshow(mfcc_feature, sr=fs, x_axis='time')
    plt.title('MFCC')
    plt.colorbar()
    filename = fn + ".png"
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()
    #
    isNan = False
    for num in mfcc_feature:
        if np.isnan(num[1]):
            isNan = True
    if isNan == False:
        write_mfcc(mfcc_feature,fn)
        #print(ceps)

#------------------------------------------------------------------------------------

print("******START*********************")

files = glob.glob("./wav/*/*.wav")

fileNum = len(files)

i = 0;
for fn in files:
    i = i + 1
    create_ceps(fn)
    if i % 10 == 0:
        print(str(i) + "/" + str(fileNum))

