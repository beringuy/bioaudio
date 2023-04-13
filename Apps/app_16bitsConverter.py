
import os
import re
from pydub import AudioSegment

def convertTo16 (filePath, destinyPath):
    name, ext = os.path.splitext(filePath)   
    tmp = re.split('([^\/]+$)', name)
    path_, file_ = list(filter(bool, tmp))    
    print('\nFile path: ', path_)
    print('File name: ', file_)
    print('File extension: ', ext, '\n')
    # Carrega o arquivo WAV de 32 bits
    audio = AudioSegment.from_wav(filePath)    
    # Converte o áudio para uma profundidade de bits de 16
    audio = audio.set_sample_width(2)
    # Salva o arquivo convertido
    newFileName = destinyPath + file_ + '_16bits' + ext
    print('aqui', newFileName)
    audio.export(newFileName, format="wav")
    return newFileName

###################3

source = '/home/alessandro/Alessandro/coughDatabaseExp/soloCough_32/'
destiny = '/home/alessandro/Alessandro/coughDatabaseExp/soloCough/'

filesList = os.listdir(source)

for file_ in filesList:
    print(source)
    print(file_)
    print(destiny)
    convertTo16(source + file_, destiny)