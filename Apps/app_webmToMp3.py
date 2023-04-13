import tensorflow as tf 
import tensorflow_io as tfio
from pydub import AudioSegment



filename = '/home/alessandro/Alessandro/audioRec.webm'

'''
file_contents = tf.io.read_file('/home/alessandro/Alessandro/teste.wav')
type(file_contents)
wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
'''

AudioSegment.from_file(filename).export("/home/alessandro/Alessandro/file.mp3", format="mp3")