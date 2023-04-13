import os
import pandas as pd
import re

import librosa

from pydub import AudioSegment
from pydub.silence import split_on_silence

##########################################

sourceFolder = '/home/ego/Datasets/COUGHVID_small/archive_wav/'

coughFileList = os.listdir(sourceFolder)

dfWhole = pd.DataFrame(coughFileList, columns =['file_name'])
dfWhole

df = dfWhole.sample(500, random_state=1) # <<<<<<<<<<<< Número de arquivos de origem
df

##########################################
# # #

def coughFinder (file):
    tmp = re.split('([^\/]+$)', file)
    path_, file_ = list(filter(bool, tmp))
    name, ext = os.path.splitext(file_)    
    print('\npath:', path_)
    print('name:', name)
    print('ext:', ext, '\n')
    if ext == '.wav':        
                
        sound = AudioSegment.from_file(file)        
        print("\n--> Original Audio length: ", len(sound), "|", sound.dBFS, " dBFS\n")
        
        chunks = split_on_silence(
            sound,
            # split on silences longer than VALUE ms (1000 [ms] = 1 sec) # ORIGINAL = 1000
            min_silence_len=750,
            # anything under -VALUE dBFS is considered silence # ORIGINAL = -16 dBFS
            silence_thresh=-26,
            # keep VALUE ms of leading/trailing silence # ORIGINAL = 200
            keep_silence=500
        )
        
        print('Concluído: split_on_silence\n')
        
        
        if len(chunks) > 0:
            i = 0
            for i in range(len(chunks)):
                print('/home/ego/Datasets/experimento_soloCough/' + file_)
                chunks[i].export('/home/ego/Datasets/experimento_soloCough/' + name + '_pt' + str(i) + ext, format="wav")    
                print(chunks[i])
                i += 1
        else:
            print('Não foi possível remover trechos de silêncio')        
    
    else:
        print('Use .wav!')
        

###########

for file_name in df['file_name']:
    tmpFile = sourceFolder + file_name
    print('\n>>> tmpFile: ', tmpFile)
    coughFinder(tmpFile)