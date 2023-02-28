# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import torch.nn as nn
import torch.nn.functional as F

#import torch.multiprocessing as mp
#from torch.utils.data.distributed import DistributedSampler
#from torch.distributed import init_process_group, destroy_process_group

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
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity


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
    
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    # parse save name
    

    print ('Loading data...')

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
     #   clip = clip.to(device)  GETS DONE in clap_.py script!
    #clip.eval()
    print ('CLAP loaded!')

    print ('Loading off-the-shelf language model...')
    import sys
    sys.path.append(args.language_model_code_path)
    from simctg import SimCTG
    #sos_token, pad_token = r'<-start_of_text->', r'<-pad->' # r'an audio clip of <-start_of_text->', r'<-pad->'
    #sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
    #sos_token = r'<-start_of_text->' # wieder weglassen
    
    generation_model = SimCTG(args.language_model_name)
    if cuda_available:
        generation_model = generation_model.to(device)
    generation_model.eval()  # set model to evaluation (test mode, i.e. no training)
    print ('Language model loaded.')
    clip_text_max_len = 77

    betas = torch.linspace(0.5, 1, steps=1).cuda()

    prompts = ["the sound of"]

    for beta in betas:
        print("Beta: " + str(beta))
        beta = beta

        for prompt in prompts:

            prompt = prompt
    

            result_list = []
            invalid_num = 0
            print ('----------------------------------------------------------------')
            with torch.no_grad():
                test_num = len(item_list)
                #test_num = 10
                print ('Number of inference instances is {}'.format(test_num))
                print ('Alpha: {0}, Beta: {1}, k: {2}'.format(args.alpha, beta, args.k))
                
                audio_sim_tables = {} # create dict to story output tables

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
                        beta, sound_instance, clip, clip_text_max_len)
                    
                    #output_text = generation_model.magic_search_gt_captions(input_ids, args.k, args.alpha, args.decoding_len, 
                     #   beta, one_test_dict['captions'], clip, clip_text_max_len) 
                    
                    last_letter_prompt = prompt[-1]
                    output_text_without_prompt = output_text.split(last_letter_prompt, 1)[1]
                    output_text_without_prompt_series = pd.Series(output_text_without_prompt)

                    one_res_dict['prediction'] = output_text_without_prompt
                    one_res_dict["beta"] = beta.item()
                    one_res_dict["prompt"] = prompt
                    one_res_dict["k"] = args.k
                    one_res_dict["alpha"] = args.alpha
                    one_res_dict["decoding_len"] = args.decoding_len
                    one_res_dict["clip_text_max_len"] = clip_text_max_len
                    one_res_dict["n_test_samples"] = test_num

                    result_list.append(one_res_dict)


                    """
                    This section produces the __output table__ in 'inference_result/similaritites_sounds' containing the:
                    1) untokenized prediction (without the prompt)
                    2) groundtruth captions
                    3) cosine similarity of the [prompt + prediction] and the [audio]
                    4) cosine similarity of the [GT_caption_i] and the [audio]
                    5) cosine similarity of the [GT captions_i] and the [prompt + prediction] (all with each other; matrix)
                    6) playable audio
                    """

                    #%% 2a) groundtruth captions

                    captions = one_res_dict["captions"] # GT captions
                    captions.append(output_text)


                    #%% 5) cosine similarity of the [GT captions_i] and the [prompt + prediction] (all with each other; matrix)

                    # NICE TO HAVE: INCLUDE ANOTHER STRING LIST ABOVE: ["G1, G2, ... P"]
                    captions_embs = clip.compute_text_representation(captions)
                    cos_sim_captions_list = cosine_similarity(captions_embs.cpu().detach().numpy()).round(2).astype(str).tolist()
                    [row.append('<br>') for row in cos_sim_captions_list]
                    cos_sim_captions_list = [val for sublist in cos_sim_captions_list for val in sublist]
                    cos_sim_captions = pd.Series({"cos_sim_captions":cos_sim_captions_list})
                    cos_sim_captions = pd.DataFrame(cos_sim_captions)[0].apply(' '.join)


                    #%% 3) 4) cosine similarity of the [GT_caption_i] and the [audio] and [prompt + prediction] and the [audio]
                    audio_embedding = clip.compute_image_representation_from_image_instance(sound_instance)
                    cos_sim = torch.cosine_similarity(audio_embedding, captions_embs)# unscaled!                   
                    cos_sim = cos_sim.cpu().detach().numpy()
                    format_string = "{:.3f}"
                    cos_sim = [format_string.format(i) for i in cos_sim.tolist()]
                    cos_sim_pred = pd.Series({"cos_sim_pred":cos_sim[-1]})
                    cos_sim = pd.DataFrame(cos_sim[:-1]).apply('<br>'.join)
            
                    #%% 6b) playable audio
                    # create col for .wav file
                    wav_col = pd.Series({"Audio":sound_full_path})                
                    sound_file_name = os.path.split(sound_full_path)[1]
                    sound_full_path = os.path.join("../../softlinks/audio_clip_test_data", sound_file_name)

                    #%% 2b) groundtruth captions
                    captions.pop()
                    captions_table = pd.Series({"captions":'<br>'.join(captions)})

                    #%% FINALIZE TABLE

                    pd.set_option('display.float_format', lambda x: '%.3f' % x)

                    cols = [output_text_without_prompt_series, captions_table, cos_sim_pred, cos_sim,  cos_sim_captions, wav_col]

                    sim_text = pd.DataFrame(pd.concat(cols, axis=0)).T

                    sim_text.columns = ["pred", "GT_captions", "sim(prompt+pred, audio)", "sim(GT_caption_i, audio)", "sim([GT_caption_i, prompt], [GT_caption_j, prompt])", "Audio"]

                    #%% 6b) playable audio
                    sim_text["Audio"] = sim_text["Audio"].apply(lambda audio_path: f"""<audio controls> <source src="{sound_full_path}" type="audio/wav"> </audio>""")
                    audio_sim_tables[str(item_list[p_idx]["sound_name"])] = sim_text

                    

                p.finish()

                #%% create table and result .json

                file_prefix = str(beta.item()) + "_" + prompt.replace(" ", "_") + "_" +"CODETEST_2_"
                html_filename =  file_prefix + "sim_audio_table.html"
                sim_audio_table = pd.concat(audio_sim_tables.values())
                html_path = os.path.join(os.getcwd(), "../inference_result/output_tables/code_testing", html_filename)
                sim_audio_table.to_html(html_path, escape=False)
            
                #print ('Inference completed!')

                save_name = args.save_name
                full_save_path = save_path_prefix + '/' +  file_prefix + save_name
                #print ('full save path is {}'.format(full_save_path))
                
                with open(full_save_path, 'w') as outfile:
                    json.dump(result_list, outfile, indent=4)
