import os
import numpy
from pydub import AudioSegment


def read(filename, limit=None):
    """
    Reads any file supported by pydub (ffmpeg) and returns the data contained
    within.

    Can be optionally limited to a certain amount of seconds from the start
    of the file by specifying the `limit` parameter. This is the amount of
    seconds from the start of the file.

    returns: (channels, samplerate)
    """
    print filename
    audiofile = AudioSegment.from_file(filename)

    if limit:
        audiofile = audiofile[:limit * 1000]

    data = numpy.fromstring(audiofile._data, numpy.int16)

    channels = []
    for i in xrange(audiofile.channels):
        channels.append(data[i::audiofile.channels])

    return channels, audiofile.frame_rate

def path_to_songname(path):
    """
    Extracts song name from a filepath. Used to identify which songs
    have already been fingerprinted on disk.
    """
    return os.path.splitext(os.path.basename(path))[0]

if __name__ == "__main__":
    channels, fs = read("C:\CloudMusic\Ken Arai - NEXT TO YOU.mp3")
    print channels, fs

