import pandas as pd
import numpy as np
import math
import os
import re
import random

import librosa
import soundfile

from pydub import AudioSegment
from pydub.silence import split_on_silence

import torchaudio
import matplotlib.pyplot as plt

#################################################

 # Em alguns testes, usar o parâmetro 'format = wav' na função AudioSegment.from_file() resultou em erros e, por isso, tal parâmetro foi removido
 # Caso erros ocorram durante a execução de alguma das funções abaixo, testar adicionar 'format = wav' e ver se resolve o problema.
 
 #################################################


'''
Recebe o caminho até um ARQUIVO e extrai para uma pasta determinada trechos do áudio original com o tamanho definido
'''
def audioSplitter (filePath, destinyFolder, step = 1000):
    
    fileName = re.search("[^/]+$", filePath)[0]
    print(fileName)
    folderPath = re.search("^.*/", filePath)[0]
    print(folderPath)    
    name, ext = os.path.splitext(fileName)
    
    os.makedirs(destinyFolder, exist_ok=True)
    
    # # # # #
    sound = AudioSegment.from_file(filePath)
    # # # # #
    # print(len(sound))    

    mainList = []

    for i in range(0, len(sound), step):
        #print('Valor de i:', i)    
        if i + step < len(sound):        
            tmpList = [i , i + step]
            mainList.append(tmpList)
        else:
            tmpList = [i , len(sound)]
            mainList.append(tmpList)    

    for clip in mainList:
        newAudio = sound[clip[0]:clip[1]]
        newAudio.export("{}{}_{}_{}.wav".format(destinyFolder, name, clip[0], clip[1]), format="wav")
    
    print ('{} splitted!'.format(fileName))
    return True

#################################################

'''
Recebe uma PASTA com os arquivos a serem convertidos e uma pasta para onde os arquivos gerados são exportados
'''
def convertToWav (sourceFolder, destinyFolder):
    tmpFolder = os.listdir(sourceFolder)
    for file in tmpFolder:
        name, ext = os.path.splitext(file)
        if ext == ".ogg" or ".webm" or ".mp3": # adicionar arquivos de audio de interesse
            sound = AudioSegment.from_file(sourceFolder + file) # checar se 'from_mp3' funciona com outros formatos de arquivo
            sound.export(destinyFolder + file + ".wav", format="wav")
            
#################################################

'''
Adiciona aos arquivos de uma PASTA, como som de fundo, arquivos aleatórios de outra pasta
'''
def addBackgroudSound (pathMain, pathBGsound,  destinyFolder):
    
    listMain = os.listdir(pathMain)
    
    listBG = os.listdir(pathBGsound)
    
    
    os.makedirs(destinyFolder, exist_ok=True)


    for audio in listMain:
        name, ext = os.path.splitext(audio)
        
        main_sound = AudioSegment.from_file(pathMain + audio)
        print('\n>>> main_sound: ', main_sound)
            
        silent_segment = AudioSegment.silent(duration=500)
        main_sound = silent_segment + main_sound + silent_segment # adiciona um trecho de áudio antes e depois do main_sound, "respiro"
        
        bg_sound = AudioSegment.from_file(pathBGsound + random.choice(listBG)) # escolhe um arquivo aleatório que será utilizado como som de fundo
        bg_sound = bg_sound + 8 # amplifica o som de fundo
        print('>>> bg_sound: ', bg_sound)
        
        final_sound = main_sound.overlay(bg_sound, loop = True) # ver na documentação parâmetro 'loop'
        
        print ('Background sound added to {}!'.format(audio))
        final_sound.export("{}{}_plusBG.wav".format(destinyFolder, name), format='wav')        
        
    return True

#################################################

'''
Recebe como input uma PASTA com arquivos .wav
Retorna um dict com informações e features dos arquivos .wav da pasta
'''
def featuresExtrator(audiosPath, n_mfcc = 13, mode_complete = False):
    listFile = []
    listExt = []
    listPath = []
    listFullPath = []    
    listSampleR = []
    
    listMfcc = []
    
    listZeroCross = []
    listSpecRoll = []
    listSpecCent = []
    listSpecBand = []  
    listRMSe = []
    listPitches = []
    listChroma = []
    
    for audio in os.listdir(audiosPath):
        name, ext = os.path.splitext(audio)
        print("\n|Name: ", name, "| & |Ext: ", ext, "|\n")
        
        if ext == '.wav':
            tmpY, tmpSampleRate = librosa.load(audiosPath + name + ext, sr=48000) # sr=22050 (default) | ‘None’ uses the native sampling rate 
                        
            listFile.append(name)
            listExt.append(ext)
            listPath.append(audiosPath)
            listFullPath.append(audiosPath + name + ext)            
            listSampleR.append(tmpSampleRate)
                                    
            ## MFCC ##
            # https://librosa.org/doc/main/generated/librosa.feature.mfcc.html#librosa.feature.mfcc            
            listMfcc.append(
                librosa.feature.mfcc(y = tmpY, sr =  tmpSampleRate, n_mfcc = n_mfcc)) # n_mfcc => number of MFCCs to return
                                                
                        
            if mode_complete == True:
                ## Zero-crossing Rate ##
                # https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html#librosa.feature.zero_crossing_rate
                listZeroCross.append(
                    librosa.feature.zero_crossing_rate(tmpY))

                ## Spectral roll-off ##
                # https://librosa.org/doc/main/generated/librosa.feature.spectral_rolloff.html#librosa.feature.spectral_rolloff
                listSpecRoll.append(
                    librosa.feature.spectral_rolloff(y = tmpY, sr = tmpSampleRate))

                ## Spectral Centroid ##
                # https://librosa.org/doc/main/generated/librosa.feature.spectral_centroid.html#librosa.feature.spectral_centroid
                listSpecCent.append(
                    librosa.feature.spectral_centroid(y = tmpY, sr = tmpSampleRate))
                
                ## Spectral Bandwidth ##
                #https://librosa.org/doc/main/generated/librosa.feature.spectral_bandwidth.html#librosa.feature.spectral_bandwidth
                listSpecBand.append(
                    librosa.feature.spectral_bandwidth(y = tmpY, sr = tmpSampleRate))

                ## RMS energy ##
                # https://librosa.org/doc/main/generated/librosa.feature.rms.html#librosa.feature.rms
                listRMSe.append(
                    librosa.feature.rms(y = tmpY))
                
                ## Pitches ##
                # https://librosa.org/doc/main/generated/librosa.piptrack.html
                pitches, magnitudes = librosa.piptrack(y = tmpY, sr = tmpSampleRate)
                listPitches.append(pitches)
                
                ## Chroma ##
                # https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html
                listChroma.append(
                    librosa.feature.chroma_stft(y = tmpY, sr = tmpSampleRate))
            
        else:
            print ("! - Use .wav")
            
        
    tmpDict =  {"file_name"          : listFile,
                    "file_format"        : listExt,
                    "path"               : listPath,
                    "full_path"          : listFullPath,                        
                    "sample_rate"        : listSampleR}       
                
    # # #
        
    listMfccMod = []
    for i in range(n_mfcc):
        listMfccMod.append([])    
        
    for mfcc_list_from_A_file in listMfcc:        
        i = 0        
        for mfcc_n_from_file in mfcc_list_from_A_file:
            listMfccMod[i].append(mfcc_n_from_file)            
            i += 1        
        
    i = 0
    for mfccNlist in listMfccMod:        
        tmpDict['mfcc_{}'.format(i+1)] = mfccNlist        
        i += 1
    
    # # #        
        
    if mode_complete == True:
        tmpDict["zero_cross_rate"] = listZeroCross
        tmpDict["spectral_rolloff"] = listSpecRoll
        tmpDict["spectral_centroid"] = listSpecCent
        tmpDict["spectral_bandwidth"] = listSpecBand
        tmpDict["root_mean_square"] = listRMSe
        tmpDict["pitches"] = listPitches
        tmpDict["chroma"] = listChroma        
    
    return tmpDict
    

#################################################

'''
Recebe como input o nome do ARQUIVO de audio, sua pasta e o destino da sua versão com trechos de silêncio removidos
Exporta os arquivos gerados para a pasta destino
Retorna "output_chunks" e a duração dos arquivos
# Checar e revisar
'''
def audioCutter (file, pathOrigin, pathDestiny):
    '''
    recorta os trechos de silêncio de um arquivo de áudio, combina os trechos e retorna o resultado junto a duração
    '''
    name, ext = os.path.splitext(file)
    if ext == '.wav':
        # Ver: https://github.com/jiaaro/pydub/issues/143 #
        print ("\n-->", file, "|", pathOrigin, "|", pathDestiny)
        
        filePath = pathOrigin + file
        sound = AudioSegment.from_file(filePath) # format="wav" removido para evitar erros
        
        print("--> Original Audio length: ", len(sound), "|", sound.dBFS, " dBFS")
        
        chunks = split_on_silence(
            sound,
            # split on silences longer than VALUE ms (1000 [ms] = 1 sec) # ORIGINAL = 1000
            min_silence_len=1000,
            # anything under -VALUE dBFS is considered silence # ORIGINAL = -16 dBFS
            silence_thresh=-36,
            # keep VALUE ms of leading/trailing silence # ORIGINAL = 200
            keep_silence=750
        )
        
        print ('chunks: ', chunks)
        
        if len(chunks) > 0:        
            finalSound = chunks[0]
            print ('chunks[0]: ', chunks[0])
        else:
            finalSound = sound
            print ('Não foi possivel usar split_on_silence. Retornando som original.')
        
        if len(chunks) > 1:        
            for i in range(len(chunks)-1):
                finalSound = finalSound + chunks[i+1]
        
        finalSound.export(pathDestiny + file, format="wav")    
        
        print("--> New Audio length: ", len(finalSound), "|", finalSound.dBFS, " dBFS")
        return finalSound, len(finalSound)
    
    else:
        print('Use .wav!')

#################################################

'''
Detectando maior duração do audio de uma determinada PASTA
Recebe como input a pasta com os arquivos de áudio
Retorna o nome do arquivo de áudio com maior duração e seu comprimento (length [len()] do AudioSegment object)
'''
def maxAudioLengthDetect (folder):
    maxAudLengthName = None
    maxAudLength = 0
    
    audioList = os.listdir(folder)
    for audio_in_file in audioList:
        name, ext = os.path.splitext(audio_in_file)
        if ext == ".wav":
            tmpPath = folder + audio_in_file
            sound = AudioSegment.from_file(tmpPath) # format="wav" removido para evitar erros       
            # https://stackoverflow.com/questions/46757852/adding-silent-frame-to-wav-file-using-python
            if len(sound) > maxAudLength:
                maxAudLengthName = audio_in_file
                maxAudLength = len(sound)
                #print(maxAudLengthName)
                #print(maxAudLength)
        else:
            print("Arquivo inválido. Use .wav")
    
    return maxAudLengthName, maxAudLength

#################################################
'''
Recebe como input a PASTA em que se deseja normatizar a duração dos audios, a pasta para onde os arquivos serão extraídos
e a duração do audio mais longo
Adiciona trechos de silêncio para que todos os arquivos fiquem com a mesma duração
Exporta os arquivos gerados para a pasta destino
'''
def makeAudiosSameLength (folder, destinyFolder, maxAudLength):
    audioForModList = os.listdir(folder)
    for audio_in_file in  audioForModList:
        name_, ext_ = os.path.splitext(audio_in_file)
        tmpPath = folder + audio_in_file
        sound = AudioSegment.from_file(tmpPath) # format="wav" removido para evitar erros
        # https://stackoverflow.com/questions/46757852/adding-silent-frame-to-wav-file-using-python
        if len(sound) < maxAudLength:
            diferenca = maxAudLength - len(sound)
            
            # create X ms of silence audio segment
            x_sec_segment = AudioSegment.silent(duration=diferenca/2)  #duration in milliseconds

            #read wav file to an audio segment
            song = AudioSegment.from_file(folder + audio_in_file)

            #Add above two audio segments    
            final_song = x_sec_segment + song + x_sec_segment
            #final_song = song # não adiciona silêncio

            #Save modified audio
            final_song.export(destinyFolder + audio_in_file, format="wav")

#################################################

'''
Extração do Mel Spectrogram de um ARQUIVO de áudio:
Recebe como input os dados na posição [0] do "mel_spectrogram" do Y do arquivo de audio, o nome do arquivo e a pasta em que o arquivo está localizado
'''
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def plot_spectrogram(spec, audioFile, folder, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  #axs.set_title(title or 'Spectrogram (db)')
  #axs.set_ylabel(ylabel)
  #axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  #fig.colorbar(im, ax=axs)
  plt.axis('off')
  #plt.show(block=False)
  plt.savefig(folder + audioFile + ".png", bbox_inches='tight', pad_inches=0)


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=None,
    hop_length=512,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=128,
    mel_scale="htk",
)

#################################################

'''
Adiciona ruído branco ao ARQUIVO de áudio:
Para ajustar o nível de ruído, modificar a variável "STD_n"
'''
# https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
# https://stackoverflow.com/questions/63898448/add-noise-to-audio-file-and-reconvert-the-noisy-signal-using-librosa-python

def addNoise(filePath, std_n = 0.025):
    fileName = re.search("[^/]+$", filePath)[0]
    folderPath = re.search("^.*/", filePath)[0]
    y, sr = librosa.load(filePath, sr=48000)

    RMS = math.sqrt(np.mean(y**2))

    STD_n = std_n # intensidade do ruído
    noise = np.random.normal(0, STD_n, y.shape[0])

    signal_noise = y + noise
    
    newFileName = re.sub("(?=\.\w*$)", ".plus_noise", fileName)

    os.makedirs(os.path.dirname(str(folderPath) + "with_noise/"), exist_ok=True) # cria uma pasta "with_noise" a partir da pasta origem do arquivo
    soundfile.write(str(folderPath) + "with_noise/" + str(newFileName), signal_noise, 48000) # exporta o arquivo para a pasta criada anteriormente
    
    return "Arquivo modificado!"