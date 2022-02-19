import cv2
import numpy as np
import scipy.signal
import timeit
import python_speech_features
from scipy.io import wavfile

from tflite_runtime.interpreter import Interpreter

# Parameters
debug_time = 0
debug_acc = 0
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 16000
resample_rate = 8000
num_channels = 1
num_mfcc = 124
model_path = 'simple_audio_model_numpy.tflite'
word_flag = 0

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)


# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)


# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs


# This gets called every 0.5 seconds
def sd_callback(rec):
    global word_flag

    # Start timing for testing
    start = timeit.default_timer()



    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)

    # Save recording onto sliding window
    #window[:len(window) // 2] = window[len(window) // 2:]
    #window[len(window) // 2:] = rec

    # Compute features
    #rec=np.resize(rec, (124,129))
    mfccs = python_speech_features.base.mfcc(rec,
                                             samplerate=new_fs,
                                             winlen=0.5,
                                             winstep=0.050,
                                             numcep=num_mfcc,
                                             nfilt=129,
                                             nfft=4048,
                                             preemph=0.0,
                                             ceplifter=0,
                                             appendEnergy=False,
                                             winfunc=np.hanning)
    mfccs = mfccs.transpose()
    mfccs=cv2.resize(mfccs,(129,124))
    #mfccs=np.expand_dims(mfccs,axis=0)
    """(im_width, im_height) = mfccs.size
    if mfccs.getdata().mode == "RGBA":
        image = mfccs.convert('RGB')
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((im_height, im_width, 3))
    mfccs=reshaped"""

    #mfccs=mfccs.reshape(129,124)

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]
    print(val)

    if debug_acc:
        print(val)

    if debug_time:
        print(timeit.default_timer() - start)


# Start streaming from microphone
rate, waveform = wavfile.read("C:/Users/ENGINEERING/Desktop/ESC-50-master/dataset/data/Others/1-12654-B-15.wav")

# run inference
sd_callback(waveform)