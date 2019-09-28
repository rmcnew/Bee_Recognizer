from scipy.io import wavfile

def load_and_scale_sound(sound_path):
    samplerate, audio = wavfile.read(sound_path)
    scaled_audio = audio/float(np.max(audio))
    return scaled_audio
