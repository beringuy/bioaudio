
# Implementing Audio Classification Project Using Deep Learning, by Raghav Agrawal
# https://www.analyticsvidhya.com/blog/2022/03/implementing-audio-classification-project-using-deep-learning/

import os
import librosa
from scipy.io import wavfile as wav
import pandas as pd
import numpy as np
from tqdm import tqdm

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

#

from BIOAUDIO.bioaudio_util import convertToWav
import shutil

############################################################################################################
# Preparo dos dados dos arquivos:

##############################################################################
# Para o projeto, todos os arquivos das diferentes pastas do dataset Urban8k #
# foram postos em uma mesma pasta, em conjunto com 1000 samples selecionados #
# aleatoriamente do dataset COUGHVID.                                        #
# Posteriormente, gravações do ambiente FIS e tosses no ambiente FIS foram   #
# adicionados.                                                               #
#                                                                            #
##############################################################################

#####################
# # # Arquivos # # #
#####################

coughPath = '/home/alessandro/Alessandro/coughDatabaseExp/soloCough/'
coughList = os.listdir(coughPath)
coughDF = pd.DataFrame(coughList, columns =['file_name'])
coughDF['class'] = 'cough'
coughDF['path'] = coughPath
coughDF

####

noCoughPath = '/home/alessandro/Alessandro/coughDatabaseExp/noCough/'
noCoughList = os.listdir(noCoughPath)
noCoughDF = pd.DataFrame(noCoughList, columns =['file_name'])
noCoughDF = noCoughDF.sample(600, random_state=1) # < < <
noCoughDF['class'] = 'noCough'
noCoughDF['path'] = noCoughPath
noCoughDF

####

coughFisPath = '/home/alessandro/Alessandro/coughDatabaseExp/experimento_soloCough_naFIS/'
coughFisList = os.listdir(coughFisPath)
coughFisDF = pd.DataFrame(coughFisList, columns =['file_name'])
coughFisDF['class'] = 'coughFis'
coughFisDF['path'] = coughFisPath
coughFisDF

################
'''
coughFisDF['class'] = 'cough'
coughDF = coughDF.sample(452, random_state=1)
df = pd.concat([coughDF, noCoughDF, coughFisDF]).reset_index(drop=True) # < < <
df['class'].value_counts()
df
'''

df = pd.concat([coughDF, noCoughDF]).reset_index(drop=True) # < < <
df['class'].value_counts()
df

############################################################################################################
# Extração dos features:

num_mfcc = 20

#
def features_extractor(file_name):    
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)    
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

#
extracted_features = []


missFile = 0
for index_num,row in tqdm(df.iterrows()):
    try:
        #file_name = sourcePath + 'archive_wav/' + row["file_name"] + '.wav'
        file_name = row['path'] + row["file_name"]
        final_class_labels = row["class"]
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])
    except FileNotFoundError:
        print("Arquivo ausente!")
        missFile +=1
    except:
        print('Erro não identificado!')
print(missFile)

extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature', 'class'])
extracted_features_df

extracted_features_df_select = extracted_features_df.loc[extracted_features_df['class'].isin(['cough','noCough'])]
extracted_features_df_select


############################################################################################################
# Preparo dos dados para o treino dos modelos:

### Split the dataset into independent and dependent dataset
X = np.array(extracted_features_df_select['feature'].tolist())
y = np.array(extracted_features_df_select['class'].tolist())

### Label Encoding -> Label Encoder
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

### Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### No of classes
num_labels = y.shape[1]

############################################################################################################

#################################
# # # # # # # MODEL # # # # # # #
#################################

model = Sequential()
### first layer
model.add(Dense(1024, input_shape = (num_mfcc, )))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
### second layer
model.add(Dense(2048))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

### third layer
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

#

### final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
model.summary()

## Training my model
num_epochs = 300
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath = './audio_classification.hdf5', 
                               verbose = 1,
                               save_best_only = True)
start = datetime.now()

#################################
############## FIT ##############
model.fit(X_train, y_train,
          batch_size = num_batch_size,
          epochs = num_epochs,
          validation_data = (X_test, y_test),
          callbacks = [checkpointer],
          verbose = 1)
#################################

duration = datetime.now() - start
print("Training completed in time: ", duration)

# Test Accuracy
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(test_accuracy[1])

#model.predict_classes(X_test)
predict_x = model.predict(X_test)
classes_x = np.argmax(predict_x, axis=1)
print(classes_x)

################################

#
# model.save('modelos/model_coughDetect_plusFIS_v5.h5') # <<<<<<<<<<
#

################################

#
# model = load_model('modelos/model_coughDetect_plusFIS_v5.h5') # <<<<<<<<<<
#

############################################################################################################
# Predict:

def predict (filePath):    
    filename = filePath
    
    #preprocess the audio file
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = num_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        
    #Reshape MFCC feature to 2-D array
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        
    #predicted_label=model.predict_classes(mfccs_scaled_features)
    x_predict = model.predict(mfccs_scaled_features) 
    predicted_label = np.argmax(x_predict, axis=1)
    print(predicted_label)    
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    print(prediction_class)
        
    return prediction_class

###########################
# Teste de predição das tosses:
num_mfcc = 20

y = np.array(df['class'].tolist())
#y = np.array(['air_conditioner','car_horn','children_playing','cough','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music'])
labelencoder = LabelEncoder()
labelencoder.fit_transform(y)
labelencoder.classes_.tolist()

########

# ARQUIVO ÚNICO:
fold = '/home/alessandro/Alessandro/coughDatabaseExp/FIS_ambient_toDetect/'
fil = 'fis07.mp3'
predict (fold + fil)

########

# PASTA:
#selectedFolder = [coughPath, 'cough']
#selectedFolder = [noCoughPath, 'noCough']
selectedFolder = [coughFisPath, 'cough']
#selectedFolder = ['/home/alessandro/Alessandro/coughDatabaseExp/construindo_FisMicRecAround/', 'noCough']

selectedClass = selectedFolder[1] # <<<<<<<<<<<<<

listFile = os.listdir(selectedFolder[0])

rightCount = 0
wrongCount = 0

for file in listFile:
    print('\nFile: ', file)
    print('Folder: ', selectedFolder[0])
    tmpFilePath = selectedFolder[0] + file
    print('File path: ', tmpFilePath, '\n')
    tmpClass = predict(tmpFilePath)
    print(file, " : ", tmpClass[0])
    if tmpClass[0] == selectedClass:
        rightCount += 1
    else:
        wrongCount += 1

print('\nRightly classified: ', rightCount, ' times')
print('Wrongly classified: ', wrongCount, ' times\n')
print ('\nACERTOS: ', rightCount, ' || ERROS: ', wrongCount, ' >> ', round(rightCount / len(listFile) * 100, 2) , '% de acerto\n')

# NOTAS: (modelo treinado com cough e não-cough):
# cough: as cough          >  r: 596  | w: 4    > 99,33% de acerto
# coughFis: as cough       >  r: 487  | w: 113  > 81,6%  de acerto
# noCough: as not cough    >  r: 1032 | w: 20   > 98,09% de acerto
# noCoughFis: as not cough >  r: 85   | w: 39   > 68,54% de acerto