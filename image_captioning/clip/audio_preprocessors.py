import importlib
torch, np = importlib.import_module('torch'), importlib.import_module('numpy')

def preprocess_for_WavCaps(sound_instance,
                           device):

    sound_instance = torch.tensor(sound_instance).unsqueeze(0).to(device)
    if sound_instance.shape[-1] < 32000 * 10:
        pad_length = 32000 * 10 - sound_instance.shape[-1]
        sound_instance = torch.nn.functional.pad(sound_instance, [0, pad_length], "constant", 0.0)

    return sound_instance


def preprocess_for_AudioCLIP(sound_instance,
                           device):

    return torch.from_numpy(sound_instance.reshape(1, -1)).to(device)


def preprocess_for_CLAP(sound_instance,
                           device):
    
    def int16_to_float32(x):
        return (x / 32767.0).astype(np.float32)


    def float32_to_int16(x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)
    
    audio_data = sound_instance.reshape(1, -1) # Make it (1,T) or (N,T)
    audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float().to(device)

    return audio_data