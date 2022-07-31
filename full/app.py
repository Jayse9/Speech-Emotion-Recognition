from flask import Flask, render_template, request
from joblib import dump, load
import pandas as pd
import numpy as np

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import sys

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
#from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


#ZCR can be interpreted as a measure of the noisiness of a signal.
def extract_features(data , sample_rate):
    #zero srossing rate
    features  = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features = np.hstack((features,zcr))
    
    #chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features = np.hstack((chroma_stft,zcr))
    
    #mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((chroma_stft,zcr))
    
    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features = np.hstack((rms,zcr))
    
    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    features = np.hstack((mel,zcr))
    
    return features

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    #data with shift
    shift_data = shift(data)
    res4 = extract_features(shift_data, sample_rate)
    result = np.vstack((result,res4))
    
    return result

basedir = os.path.abspath(os.path.dirname(__file__))

# model=pickle.load(open(basedir+'\saved_models\Emotion_Voice_Detection_Model.h5','rb'))
# print ("model loaded")



# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model_final.h5")
print("Loaded model from disk")    

app =Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] =1
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after',methods=['GET', 'POST'])
def after():
    global loaded_model 
    X=[]
    file = request.files['file1']
    file.save("static/file.wav")
    
    features = get_features("static/file.wav")
    for ele in features:
        X.append(ele)
    
    Input = pd.DataFrame(X)    
    scaler = StandardScaler()
    Input = scaler.fit_transform(Input)
    Input = np.expand_dims(Input, axis=2)
    
    Output = loaded_model.predict(Input)
    encoder1 = load('encoder.joblib')
    final =encoder1.inverse_transform(Output)
    final = final.flatten()
    print(final)
    return render_template('predict.html',final = final[0])
    

if __name__ == '__main__':
    app.run(debug=True)