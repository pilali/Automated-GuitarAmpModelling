from CoreAudioML.dataset import audio_converter, audio_splitter
from scipy.io import wavfile

filename="./Data/train/ht1-input"

audio = wavfile.read(filename+'.wav')

raw_audio = audio_converter(audio[1])

splitted_audio = audio_splitter(raw_audio, [0.25, 0.25, 0.25, 0.25])

count = 0
for stem in splitted_audio:
    wavfile.write(filename+'_'+str(count)+'.wav', 44100, stem)
    count=count+1
