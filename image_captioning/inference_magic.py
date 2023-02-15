# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar
from PIL import Image
import librosa
import sys
import pandas as pd

import logging

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

logging.getLogger('transformers.generation_utils').disabled = True

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # language model path setting
    parser.add_argument("--language_model_code_path", type=str, help="where is the code of language model located")
    # model and data configuration
    parser.add_argument("--language_model_name", type=str, help="name of pre-trained language model")
    # index path setting
    parser.add_argument("--clap_path", type=str, help="where is the clip code located")
    parser.add_argument("--clap_model_name", type=str)
    parser.add_argument("--test_image_prefix_path", type=str, help="the folder that stores all test images")
    # test data path
    parser.add_argument("--test_path", type=str)
    # decoding configuration
    parser.add_argument("--decoding_len", type=int, default=16, help='maximum length of (prefix + generated continuation)')
    # sample rate 
    parser.add_argument("--sample_rate", type=int, default=44100)
    # magic configuration
    parser.add_argument("--k", type=int, default=-1, help='k for magic search')
    parser.add_argument("--alpha", type=float, default=-1.0, help="alpha for magic search")
    parser.add_argument("--beta", type=float, default=-1.0, help="beta for magic search")
    # save configuration
    parser.add_argument("--save_path_prefix", type=str, help="save the result in which directory")
    parser.add_argument("--save_name", type=str, help="the name of the saved file")
    return parser.parse_args()

def get_prompt_id(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").input_ids   # gets token id's for the text: e.g. hello I am cool [50257, 281, 6597, 10651, 286, 257]
    return tokens

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    save_path_prefix = args.save_path_prefix
    import os
    
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    # parse save name
    save_name = args.save_name
    full_save_path = save_path_prefix + '/' + save_name
    print ('full save path is {}'.format(full_save_path))

    print ('Loading data...')
    import json

    with open(args.test_path) as f:
        item_list = json.load(f)
    print ('Data loaded.')
    print ('Number of test instances is {}'.format(len(item_list)))
    
    # get AudioCLIP
    import sys
    sys.path.append(args.clap_path)  # define path to clap class

    print ('Loading CLAP...')

    seed = 4182
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    from clap_ import CLIP
    clip = CLIP(args.clap_model_name) #ACLP *.pt # er soll hier CLIP aus clip.py laden (dort wird AudioCLIP geladen)
    #if cuda_available:
     #   clip = clip.cuda(device)
    #clip.eval()
    print ('CLAP loaded!')

    print ('Loading off-the-shelf language model...')
    import sys
    sys.path.append(args.language_model_code_path)
    from simctg import SimCTG
    #sos_token, pad_token = r'<-start_of_text->', r'<-pad->' # r'an audio clip of <-start_of_text->', r'<-pad->'
    #sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    #sos_token = r'<-start_of_text->' # wieder weglassen
    prompt = "This is an audio clip of"
    clip_text_max_len = 77
    generation_model = SimCTG(args.language_model_name)
    if cuda_available:
        generation_model = generation_model.to(device)
    generation_model.eval()  # set model to evaluation (test mode, i.e. no training)
    print ('Language model loaded.')

    result_list = []
    invalid_num = 0
    print ('----------------------------------------------------------------')
    with torch.no_grad():
        test_num = len(item_list)
        #test_num = 10
        print ('Number of inference instances is {}'.format(test_num))
        print ('Alpha: {0}, Beta: {1}, k: {2}'.format(args.alpha, args.beta, args.k))
        p = progressbar.ProgressBar(test_num)
        p.start()
        for p_idx in range(test_num):
            p.update(p_idx)
            one_test_dict = item_list[p_idx]

            one_res_dict = {
                'split':one_test_dict['split'],
                'sound_name':one_test_dict['sound_name'],
                #'file_path':one_test_dict['file_path'],
                'captions':one_test_dict['captions']  
            }

            sound_full_path = args.test_image_prefix_path + '/' + one_test_dict['sound_name']

            #print(one_test_dict['sound_name'])
            # create sound instance 
            sound_instance, _ = librosa.load(sound_full_path, sr=args.sample_rate)
            
            # tokenize 
            input_ids = get_prompt_id(prompt, generation_model.tokenizer) 
            
            """
            input ids: vector containing tokens of prompt [50257, 271, 6597, 10651, 286, 257]
            """

            if cuda_available:
                input_ids = input_ids.cuda(device)

            #try:

            """
            input_ids: gets token id of the SOS token 
            """

            output_text = generation_model.magic_search(input_ids, args.k, args.alpha, args.decoding_len, 
                args.beta, sound_instance, clip, clip_text_max_len)

            one_res_dict['prediction'] = output_text
            result_list.append(one_res_dict)

            # Produce output table
            
            audio_embedding = clip.compute_image_representation_from_image_instance(sound_instance)
            captions = one_res_dict["captions"] # GT captions
            captions.append(output_text)

            # get unsoftmaxed cos sims

            captions_embs = clip.compute_text_representation(captions)
            cos_sim = torch.cosine_similarity(audio_embedding, captions_embs) # unscaled! 

            # get softmax cos sims

            cos_sim_softmax = clip.compute_image_text_similarity_via_raw_text(image_embeds=audio_embedding, text_list=captions)
            
            # create col for .wav file
            wav_col = pd.Series([sound_full_path] * len(captions))
            
            # create absolute path

            #root_dir = os.path.join(os.getcwd(), "../")
            sound_file_name = os.path.split(sound_full_path)[1]
            sound_full_path = os.path.join("../../softlinks/audio_clip_test_data", sound_file_name)
            print("sound_full_path: ")
            print(sound_full_path)
            cols = [pd.Series(cos_sim_softmax.flatten().cpu().detach().numpy().T), pd.Series(cos_sim.cpu().detach().numpy()), pd.Series(captions), wav_col]

            sim_text = pd.concat(cols, axis=1)
            sim_text.columns = ["softmaxed_cos_sim", "cos_sim", "Captions", "Audio"]

            sim_text["Audio"] = sim_text["Audio"].apply(lambda audio_path: f"""<audio controls> <source src="{sound_full_path}" type="audio/wav"> </audio>""")

            html_filename =  os.path.split(sound_full_path)[-1] + ".html"
            
            html_path = os.path.join(os.getcwd(), "../inference_result/similarities_sounds", html_filename)
            sim_text.to_html(html_path, escape=False)
            

        p.finish()
    print ('Inference completed!')

    import json
    with open(full_save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
