import torch
import requests
from torch import nn
import sys
import os
import torch
import librosa
import glob
import pandas as pd
import torchaudio
import torchvision
import numpy as np

from CLAP.src.open_clip import create_model
#from CLAP.src.training.data import get_audio_features
#from CLAP.src.training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer

def get_mel(audio_data,audio_cfg):
    # mel shape: (n_mels, T)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    )(audio_data)
    # Align to librosa:
    # librosa_melspec = librosa.feature.melspectrogram(
    #     waveform,
    #     sr=audio_cfg['sample_rate'],
    #     n_fft=audio_cfg['window_size'],
    #     hop_length=audio_cfg['hop_size'],
    #     win_length=audio_cfg['window_size'],
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=64,
    #     norm=None,
    #     htk=True,
    #     f_min=audio_cfg['fmin'],
    #     f_max=audio_cfg['fmax']
    # )
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    """
    with torch.no_grad():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size']+1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames-chunk_frames+1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front+chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle+chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back+chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len/len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len/len(audio_data))
                    audio_data = audio_data.repeat(n_repeat+1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)



class CLIP():

    def __init__(self, model_name):

        """
        Loads the specified text and audio model (.pt files) in text_model_name and audio_model_name. 
        This name has to be specified as a flag in the .sh script.
        
        """
        # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90 
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        print ('CLAP tokenizer initialized')
    

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        precision = 'fp32'
        amodel = 'HTSAT-tiny' # or 'PANN-14'
        tmodel = 'roberta' # the best text encoder in our training
        enable_fusion = False # False if you do not want to use the fusion model
        fusion_type = 'aff_2d'
        pretrained = model_name # the checkpoint name, the unfusion model can also be loaded.

        self.a_model, self.a_model_cfg = create_model(
            amodel,
            tmodel,
            pretrained,
            precision=precision,
            device=device,
            enable_fusion=False,
            fusion_type=fusion_type
        )

        print ('Initialized CLAP Audio Model')

        self.a_model.eval()

        print( 'Turned on eval mode')

        self.logit_scale_a, self.logit_scale_t = self.a_model(None, None, device)
        self.logit_scale_a = self.logit_scale_a.cpu()

        torch.cuda.empty_cache()
        print ('Cuda cache emptied')



    @torch.no_grad()
    def compute_image_representation_from_image_path(self, audio_file:str, sample_rate:int):

        """
        Loads the specified model (.pt file) in model_name. This name has to be specified as a flag in the .sh script.
        Creates an embedding vector for the specified audio file (.wav).
        Sample rate that is utilized to sample the audio file.
        """

        # load the waveform of the shape (T,), should resample to 48000

        audio_waveform, sr = librosa.load(audio_file, sr=sample_rate) 

        # quantize
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        audio_dict = {}

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        audio_dict = get_audio_features(
            audio_dict, audio_waveform, 480000, 
            data_truncating='rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=self.a_model_cfg['audio_cfg']
        )
        # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
        audio_embed = self.a_model.get_audio_embedding([audio_dict])

        return audio_embed, audio_file


    def compute_image_representation_from_image_instance(self, sound_instance):

         # quantize
        audio_waveform = int16_to_float32(float32_to_int16(sound_instance))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        audio_dict = {}

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        audio_dict = get_audio_features(
            audio_dict, audio_waveform, 480000, 
            data_truncating='rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=self.a_model_cfg['audio_cfg']
        )
        # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
        
        audio_embed = self.a_model.get_audio_embedding([audio_dict])

        return audio_embed


    def compute_text_representation(self, text_data:list):
        
        """
        Creates an embedding vector for every element of text_data. 
        """

        # load the text, can be a list (i.e. batch size)

        def tokenizer(text):
            result = self.tokenize(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in result.items()}    

        text_data = tokenizer(text_data)
        
        text_embed = self.a_model.get_text_embedding(text_data)

        return text_embed


    def compute_image_text_similarity_via_embeddings(self, image_embeds, text_embeds):
        '''
            image_embeds: 1 x embed_dim
            text_embeds: len(text_list) x embed_dim
        '''
        # CLAP, but text_embed's norm != 1 
        #logits_per_audio = self.logit_scale_a * image_embeds @ text_embeds.t().detach().cpu()
        
        # Mine like MAGIC

        logits_per_text = self.logit_scale_a * torch.cosine_similarity(image_embeds, text_embeds) 
        logits_per_image = torch.unsqueeze(logits_per_text.T, 0)
        

        return logits_per_image.softmax(dim=-1)  # 1 x len(text_list)


    def compute_image_text_similarity_via_raw_text(self, image_embeds, text_list):

        text_embeds = self.compute_text_representation(text_list)

        return self.compute_image_text_similarity_via_embeddings(image_embeds, text_embeds)
        

    ### -------------------- functions for building index ---------------------- ###
    def compute_batch_index_image_features(self, audio_list):
        '''
            # list of sound instances
        '''
        audio_dicts = []

        for audio in audio_list:

            audio_waveform = int16_to_float32(float32_to_int16(audio))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            audio_dict = {}

            audio_dict = get_audio_features(
                audio_dict, audio_waveform, 480000, 
                data_truncating='rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.a_model_cfg['audio_cfg']
            )

            audio_dicts.append(audio_dict)

    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
        audio_embeds = self.a_model.get_audio_embedding(audio_dicts)

        return audio_embeds #len(sound_list) x embed_dim IS THAT TRUE?


    def compute_batch_index_text_representation(self, text_list):

        # text_list: a list of text

        return [self.compute_text_representation for sample in text_list] # batch_size x n_captions_per_track x embed_dim IS THAT TRUE?


