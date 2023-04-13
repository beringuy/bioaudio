from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import numpy as np
from PIL import Image

from bioaudio_util import maxAudioLengthDetect, makeAudiosSameLength, audioCutter, plot_spectrogram, mel_spectrogram

import torchaudio
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import librosa
import soundfile as sf
import noisereduce as nr


# # # # # # # # #
### BIOAUDIO ###

modelCough = load_model('./model_coughDetect_plusFIS_v8_mfcc40_multi.h5')

modelCovid = load_model('./model_64percent_v1.h5')

# Redução de Ruído:
def noiseReduct(sourceFolder):
    print('\n>> Iniciando: noiseReduct')
    y, sr = librosa.load(sourceFolder + 'tmpAudioForAnalysis.wav', sr=48000)    
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease = 0.8)    
    sf.write(sourceFolder + "postNoiseRedct.wav", reduced_noise, sr, format='wav')
    print('>> Concluído: noiseReduct\n')
    
    return "Ruído reduzido"


# Extração de espectrograma
def silence_spectro_extract (sourceFolder, singleFile): ### rever aplicação de silencio e spec na mesm a função <<<
    print('\n>> Iniciando: silence_spectro_extract')
    # Cria a pasta para armazenar o arquivo gerado (silêncio removido), caso não exista:
    os.makedirs(os.path.dirname(sourceFolder + "silenced/"), exist_ok=True)
    # Remove trechos de silêncio:    
    tmp, tmpLength = audioCutter(singleFile, sourceFolder, sourceFolder + "silenced/")
    # Carrega arquivo gerado:    
    tmp_data_waveform, tmp_rate_of_sample = torchaudio.load(sourceFolder + singleFile)
    # Gera e extrai espectrograma:
    melspec = mel_spectrogram(tmp_data_waveform)
    plot_spectrogram(melspec[0], singleFile, sourceFolder)
    print('>> Concluído: silence_spectro_extract\n')

    return "Espectrograma extraído"


# Pré-processamento
def img_pre_process(imagePath, img_width, img_height):
    print('\n>> Iniciando: img_pre_process')
    # Carrega a imagem, ajusta o tamanho e a escala de cor e a transforma em array:    
    img = tf.keras.utils.load_img(imagePath, target_size = (img_width, img_height), color_mode="grayscale")
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    print('>> Concluído: img_pre_process\n')
    
    return img


# Resultados das avaliações:
def modelResult (img, sourceFolder):
    print('\n>> Iniciando: modelResult')
    
    if predict(sourceFolder + 'tmpAudioForAnalysis.wav')[0] != 'cough' and predict(sourceFolder + 'tmpAudioForAnalysis.wav')[0] != 'coughFis':        
        return ["O sistema NÃO identificou o input como uma tosse!", None, None]
    else:        
        sig01_covid = round(modelCovid.predict(img)[0][0]*100, 2)        
        class_ = (modelCovid.predict(img) > 0.5).astype("int32")[0][0]
        print ('sig01_covid : ', sig01_covid)
        print ('class_ : ', class_, ' >> 0:covid // 1:saude')
        
        if sig01_covid <= 10:
            return ["O sistema identificou o input como uma tosse!", "COVID-19: Grau 2", sig01_covid]
            
        elif sig01_covid > 10 and sig01_covid <= 30:
            return ["O sistema identificou o input como uma tosse!", "COVID-19: Grau 1", sig01_covid]
            
        elif sig01_covid > 30 and sig01_covid <= 55:
            return ["O sistema identificou o input como uma tosse!", "Estado de saúde indeterminado", sig01_covid]
            
        elif sig01_covid > 55 and sig01_covid <= 75:
            return ["O sistema identificou o input como uma tosse!", "Saudável: Grau 1", sig01_covid]
            
        elif sig01_covid > 75 and sig01_covid <= 90:
            return ["O sistema identificou o input como uma tosse!", "Saudável: Grau 2", sig01_covid]
            
        elif sig01_covid > 90:
            return ["O sistema identificou o input como uma tosse!", "Saudável: Grau 3", sig01_covid]
        
        else:
            print('ERRO NA CLASSIFICAÇÃO DE COVID!')
            
        print('\n>> Concluído: modelResult\n')
            
        
# Predição do modelo da tosse baseado em audio features:
num_mfcc = 40

y = np.array(['air_conditioner', 'car_horn', 'children_playing', 'cough', 'coughFis', 'dog_bark', 'drilling', 'engine_idling', 'fisAround', 'gun_shot', 'jackhammer', 'siren', 'street_music']) ### <<<<<<<<<<<<<<<<<<<<<
labelencoder = LabelEncoder()
labelencoder.fit_transform(y)
#labelencoder.classes_

def predict (filePath):
    print('\n>> Iniciando: predict')    
    
    #preprocess the audio file / mfcc extract
    audio, sample_rate = librosa.load(filePath, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = num_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    #Reshape MFCC feature to 2-D array
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    
    #predicted_label=model.predict_classes(mfccs_scaled_features) ### <<<
    x_predict = modelCough.predict(mfccs_scaled_features)
    print(' > x_predict:', x_predict)    
    predicted_label = np.argmax(x_predict, axis=1)        
    print(' > label:', predicted_label)    
    prediction_class = labelencoder.inverse_transform(predicted_label)    
    print(' > Class:', prediction_class)
    print('>> Concluído: predict\n')
    
    return prediction_class

def coughPredict(sourceFolder):    
    print('\n>> Iniciando: coughPredict')
    fileFolder = os.listdir(sourceFolder) # garantir que somente o arquivo correto passe:
    
    for singleFile in fileFolder:        
        name, ext = os.path.splitext(singleFile)
        if name == 'tmpAudioForAnalysis' and ext == ".wav":
            try:                
                predict(sourceFolder + singleFile)                
                print('>> Concluído: coughPredict\n')
                return True
            except:
                print ('>> Concluído com erro: coughPredict\n')
                return False


# Predição
def singleFileToPredict(sourceFolder):
    print('\n>> Iniciando: singleFileToPredict')
    fileFolder = os.listdir(sourceFolder)
        
    for singleFile in fileFolder:
        name, ext = os.path.splitext(singleFile)
        if name == 'tmpAudioForAnalysis' and ext == ".wav":
            noiseReduct(sourceFolder)  ### <<<
            
            if coughPredict(sourceFolder) == True:
                # Extrai espectrograma:
                silence_spectro_extract(sourceFolder, 'tmpAudioForAnalysis.wav')  ### <<<
                silence_spectro_extract(sourceFolder, 'postNoiseRedct.wav')  ### <<<
    
                # Carrega a imagem, ajusta o tamanho e a escala de cor e a transforma em array:
                imgPath = sourceFolder + 'tmpAudioForAnalysis.wav' + ".png" ### <<<
                #imgPath = sourceFolder + 'postNoiseRedct.wav' + ".png" ### <<<
                img = img_pre_process(imgPath, 496, 369)                
                # Usa a imagem gerada como input para os modelos e exibe o resultado:
                result = modelResult(img, sourceFolder)
                print('\n>> Concluído: singleFileToPredict\n')
                return result
            
            else:
                print('\n>> Concluído: singleFileToPredict\n')
                return ["O sistema NÃO identificou o input como uma tosse!", None, None]
        
        

# # # # # # # # # #

app = Flask(__name__)
CORS(app)


@app.route('/upload-audio', methods=["GET", "POST"])
def test():    
    audioTeste = request.files
    print ("\n\n========================================================================")
    print ("========================================================================")
    print ("\nINÍCIO...")
    print ("\nSUCESSO! - áudio recebido")
    
    file_storage = audioTeste["audio"]    

    sourceFolder = './forAnalysis/'  ### <<<
    arquivo = sourceFolder + 'tmpAudioForAnalysis.wav'
    
    if os.path.exists(arquivo):
        os.remove(arquivo)
        print(f"\nArquivo {arquivo} removido com sucesso!\n")
    else:
        print(f"\nArquivo {arquivo} inexistente.\n")

    os.makedirs(os.path.dirname(sourceFolder), exist_ok=True)
    file_storage.save(sourceFolder + "tmpAudioForAnalysis.wav")
        
    # # #    
    resultado = singleFileToPredict(sourceFolder)
    print(resultado)
    print ("\nPROCESSO CONCLUÍDO!")
    print ("\n========================================================================")
    print ("========================================================================")
    # # #
    
    response = {
            'prediction': {
                'cough': resultado[0],
                'covid': resultado[1],
                'sig01_covid' : resultado[2],
            }
        }
    
    return jsonify(response), 200
    

# # # # # # # # # #

if __name__ == '__main__':
    app.run()
