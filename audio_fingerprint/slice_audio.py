# -*- coding: utf-8 -*
'''
get sliced audio files
'''

import numpy as np
from pydub import AudioSegment
import os

WORK_PATH = 'C:/audio depository'
DIR_PATH = 'C:/audio depository/sample_slice'
os.chdir(WORK_PATH)
file_list = os.listdir('./')
for file_ in file_list:
    filename, extension = os.path.splitext(file_)
    if extension == '.mp3':
        song = AudioSegment.from_file(file_)
        a = np.random.rand()
        b = np.random.rand()
        a = 0.3
        b = 0.35
        length = len(song)
        lower_bound = round(min(a, b) * length)
        upper_bound = round(max(a, b) * length)
        song_slice = song[lower_bound: upper_bound]
        song_slice.export('%s/%s_slice.mp3' % (DIR_PATH, filename), format="mp3")

