import cv2
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from scipy import signal
import time

from tflite_runtime.interpreter import Interpreter





VERBOSE_DEBUG = True


def print_info(waveform):
    # audio data
    if VERBOSE_DEBUG:
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])


def show_audio(wavfile_name):
    # get audio data
    rate, waveform0 = wavfile.read(wavfile_name)

    print_info(waveform0)

    # if stereo, pick the left channel
    waveform = None
    if len(waveform0.shape) == 2:
        print("Stereo detected. Picking one channel.")
        waveform = waveform0.T[1]
    else:
        waveform = waveform0

        # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    display.display(display.Audio(waveform, rate=16000))

    print("signal max: %f RMS: %f abs: %f " % (np.max(waveform),
                                               np.sqrt(np.mean(waveform ** 2)),
                                               np.mean(np.abs(waveform))))

    max_index = np.argmax(waveform)
    print("max_index = ", max_index)

    fig, axes = plt.subplots(4, figsize=(10, 8))

    timescale = np.arange(waveform0.shape[0])
    axes[0].plot(timescale, waveform0)

    timescale = np.arange(waveform.shape[0])
    axes[1].plot(timescale, waveform)

    # scale and center
    waveform = 2.0 * (waveform - np.min(waveform)) / np.ptp(waveform) - 1

    timescale = np.arange(waveform.shape[0])
    axes[2].plot(timescale, waveform)

    timescale = np.arange(16000)
    start_index = max(0, max_index - 8000)
    end_index = min(max_index + 8000, waveform.shape[0])
    axes[3].plot(timescale, waveform[start_index:end_index])

    plt.show()


def process_audio_data(waveform):
    """Process audio input.
    This function takes in raw audio data from a WAV file and does scaling
    and padding to 16000 length.
    """

    if VERBOSE_DEBUG:
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # if stereo, pick the left channel
    if len(waveform.shape) == 2:
        print("Stereo detected. Picking one channel.")
        waveform = waveform.T[1]
    else:
        waveform = waveform

    if VERBOSE_DEBUG:
        print("After scaling:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)
    print("peak-to-peak: %.4f. Adjust as needed." % (PTP,))

    # return None if too silent
    if PTP < 0.5:
        return []

    if VERBOSE_DEBUG:
        print("After normalisation:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # scale and center
    waveform = 2.0 * (waveform - np.min(waveform)) / PTP - 1

    # extract 16000 len (1 second) of data
    max_index = np.argmax(waveform)
    start_index = max(0, max_index - 40)
    end_index = min(max_index + 40, waveform.shape[0])
    waveform = waveform[start_index:end_index]

    # Padding for files with less than 16000 samples
    if VERBOSE_DEBUG:
        print("After padding:")

    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform

    if VERBOSE_DEBUG:
        print("waveform_padded:", waveform_padded.shape, waveform_padded.dtype, type(waveform_padded))
        print(waveform_padded[:5])

    return waveform_padded


def get_spectrogram(waveform):

    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # compute spectrogram
    f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255,
        noverlap = 124, nfft=256)
    # Output is complex, so take abs value
    out=np.resize(Zxx,(124,129))
    output=cv2.resize(out,(124,129))

    spectrogram = np.abs(output)
    #spectrogram = np.resize(spectrogram, (129, 1685)).astype('float32')
    #spectrogram.set_shape((129, 1685))

    if VERBOSE_DEBUG:
        print("spectrogram:", spectrogram.shape, type(spectrogram))
        print(spectrogram[0, 0])

    return spectrogram


def run_inference(waveform):
    # get spectrogram data
    spectrogram = get_spectrogram(waveform)

    if not len(spectrogram):
        # disp.show_txt(0, 0, "Silent. Skip...", True)
        print("Too silent. Skipping...")
        # time.sleep(1)
        return

    spectrogram1 = np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))

    if VERBOSE_DEBUG:
        print("spectrogram1: %s, %s, %s" % (type(spectrogram1), spectrogram1.dtype, spectrogram1.shape))

    # load TF Lite model
    interpreter = Interpreter('simple_audio_model_numpy.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print(input_details)
    # print(output_details)

    input_shape = input_details[0]['shape']
    input_data = spectrogram1.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("running inference...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    yvals = output_data[0]
    commands = ['others', 'chainsaw']

    if VERBOSE_DEBUG:
        print(output_data[0])
    print(output_data[0])
    # print(">>> " + commands[np.argmax(output_data[0])].upper())
    # disp.show_txt(0, 12, commands[np.argmax(output_data[0])].upper(), True)
    # time.sleep(1)


def main():
    # create parser

    """parser = argparse.ArgumentParser(description=descStr)
    # add a mutually exclusive group of arguments
    group = parser.add_mutually_exclusive_group()

    # add expected arguments
    group .add_argument('--input', dest='wavfile_name', required=False)

    # parse args
    args = parser.parse_args()

    """

    # test WAV file

    # get audio data
    rate, waveform = wavfile.read("C:/Users/ENGINEERING/Desktop/ESC-50-master/dataset/data/Chain_saw/1-116765-A-41.wav")
    # run inference
    run_inference(waveform)

    print("done.")


# main method
if __name__ == '__main__':
    # record('sound.wav')
    main()

    """for i in range(0,5,1):
        record('first.wav')
        time.sleep(0.5);
        main()"""