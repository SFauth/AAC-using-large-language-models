# coding=utf-8
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

from evaluation.pycocoevalcap.eval import COCOEvalCap_obs
from evaluation.pycocoevalcap.eval import COCOEvalCap_list

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
    parser.add_argument("--save_name", type=str, help="the name of the saved file")
    parser.add_argument("--experiment", type=str, help="specify: hyperparam_experiments or code_testing")
    # data set
    parser.add_argument("--dataset", type=str, help="specify dataset: clotho or AudioCaps")
    # prompt
    parser.add_argument("--include_prompt_magic", type=str, help="include prompt in the calculation of the MAGIC score")

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

    print ('Loading data...')

    with open(args.test_path) as f:
        item_list = json.load(f)
    print ('Data loaded.')

    #item_list = item_list[0:3]

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

    #item_list = item_list[0:3]

    betas = torch.linspace(0.5, 4, steps=11).cuda()
    #betas_2 = torch.linspace(0, 0.5, steps=11).cuda()
    #betas = torch.concat([betas_1, betas_2]).unique()
    #betas = torch.linspace(1, 2, steps=11).cuda()
    prompts = ["The sound of" ,"This is a sound of", "This is the sound of"]

    temperatures = torch.linspace(35, 42.5, steps=6).cuda()
    #temperatures = torch.linspace(42.5, 50, steps=6).cuda()

    #betas = torch.tensor([0.15], device="cuda")
    #prompts = ["this is the sound of"]

    for temperature in temperatures:
        print("Temperature: " + str(temperature))
        clip.logit_scale_a = temperature.unsqueeze(dim=0)

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

                    var_magic_scores_list = []
                    var_model_conf_list = []

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
                        
                        sound_full_path = args.test_image_prefix_path + one_test_dict['sound_name']

                        # create sound instance 
                        try:
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

                            output_text, var_magic_scores, var_model_conf = generation_model.magic_search(input_ids, args.k, args.alpha, args.decoding_len, 
                                beta, sound_instance, clip, clip_text_max_len, args.include_prompt_magic)
                            
                            #output_text, var_magic_scores, var_model_conf = generation_model.magic_search_gt_captions(input_ids, args.k, args.alpha, args.decoding_len, 
                                #   beta, one_test_dict['captions'], clip, clip_text_max_len,  args.include_prompt_magic) 
                            
                            var_magic_scores_list.append(var_magic_scores)
                            var_model_conf_list.append(var_model_conf)

                            last_letter_prompt = prompt[-1]
                            output_text_series = pd.Series(output_text)
                            output_text_without_prompt = output_text.split(last_letter_prompt, 1)[1]
                            output_text_without_prompt_series = pd.Series(output_text_without_prompt)

                            one_res_dict['prediction'] = output_text_without_prompt # always without prompt, as prompt is other entry
                            one_res_dict["beta"] = beta.item()
                            one_res_dict["prompt"] = prompt
                            one_res_dict["k"] = args.k
                            one_res_dict["alpha"] = args.alpha
                            one_res_dict["decoding_len"] = args.decoding_len
                            one_res_dict["clip_text_max_len"] = clip_text_max_len
                            one_res_dict["n_test_samples"] = test_num
                            one_res_dict["included_prompt_in_magic"] = args.include_prompt_magic
                            one_res_dict["dataset"] = args.dataset
                            one_res_dict["CLAP_type"] = os.path.split(args.clap_model_name)[-1]
                            one_res_dict["temperature"] = temperature.item()

                            result_list.append(one_res_dict)


                            """
                            This section produces the __output table__ in 'inference_result/similaritites_sounds' containing the:
                            1) untokenized prediction (without the prompt)
                            2) groundtruth captions
                            3) cosine similarity of the [prediction] and the [audio]
                            4) cosine similarity of the [GT_caption_i] and the [audio]
                            5) cosine similarity of the [GT captions_i] and the [prediction] (all with each other; matrix)
                            6) playable audio
                            7) metrics for the current observation or the whole run (first row)
                            DISCLAIMER: the prediction in 3) and 5) contain the prompt, if include_magic_prompt == True
                            """

                            #%% 2a) groundtruth captions and prediction

                            captions = one_res_dict["captions"] # GT captions
                            
                            if args.include_prompt_magic == "True":
                                captions.append(output_text)
                                pred = output_text_series
                                table_subfolder = "includes_prompt_magic"
                            
                            else: 
                                captions.append(output_text_without_prompt)
                                pred = output_text_without_prompt_series
                                table_subfolder = "excludes_prompt_magic"


                            #%% 5) cosine similarity of the [GT captions_i] and the [prediction] (all with each other; matrix)
                            # includes prompt if specified in flag
                            # CHECK THIS COLUUUUUUUUUUUUUUUUUMN IN TABLE AND ITS CAPTION !!
                            # NICE TO HAVE: INCLUDE ANOTHER STRING LIST ABOVE: ["G1, G2, ... P"]
                            captions_embs = clip.compute_text_representation(captions)
                            cos_sim_captions_list = cosine_similarity(captions_embs.cpu().detach().numpy()).round(2).astype(str).tolist()
                            [row.append('<br>') for row in cos_sim_captions_list]
                            cos_sim_captions_list = [val for sublist in cos_sim_captions_list for val in sublist]
                            cos_sim_captions = pd.Series({"cos_sim_captions":cos_sim_captions_list})
                            cos_sim_captions = pd.DataFrame(cos_sim_captions)[0].apply(' '.join)


                            #%% 3) 4) cosine similarity of the [GT_caption_i] and the [audio] and [prediction] and the [audio]
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

                            if args.dataset == "clotho":
                                sound_full_path = os.path.join("../../../../../softlinks_to_wav/evaluation_data_files", sound_file_name)

                            elif args.dataset == "audiocaps":
                                sound_full_path = os.path.join("../../../../../softlinks_to_wav/AudioCaps_data", sound_file_name)

                            else:
                                pass

                            if args.language_model_name == "facebook/opt-1.3b":
                                sound_full_path = os.path.join("..", sound_full_path)

                            #%% 2b) groundtruth captions
                            captions.pop()
                            captions_table = pd.Series({"captions":'<br>'.join(captions)})

                            #%% 7) metrics

                            # compute the metrics

                            # expects results json (a list of dicts)DDDDDDDDDDDDDDDDDDd
                            """
                            cocoEval = COCOEvalCap_obs(one_res_dict=one_res_dict)
                            cocoEval.evaluate()
                            metrics = pd.DataFrame(cocoEval.metrics).apply(lambda x: x.round(2), axis=0)
                            metrics = metrics.fillna(0).apply(lambda x: x.sum()).to_frame().T
                            """

                            #%% COMBINE TABLE COMPONENTS

                            pd.set_option('display.float_format', lambda x: '%.3f' % x)

                            cols = [pred, captions_table, cos_sim_pred, cos_sim,  cos_sim_captions, wav_col]

                            sim_text = pd.DataFrame(pd.concat(cols, axis=0)).T

                            sim_text.columns = ["pred", "GT_captions", "sim(pred, audio)", "sim(GT_caption_i, audio)", "sim(GT_caption_i, GT_caption_j, pred]) , while j!=i", "Audio"]

                            #%% 6b) playable audio
                            sim_text["Audio"] = sim_text["Audio"].apply(lambda audio_path: f"""<audio controls> <source src="{sound_full_path}" type="audio/wav"> </audio>""")

                            #sim_text = pd.concat([sim_text, metrics], axis=1)

                            audio_sim_tables[str(item_list[p_idx]["sound_name"])] = sim_text

                                
                            
                        except: 
                           next

                    p.finish()

                    #%% add NLG metrics for whole run to table

                    cocoEval_final = COCOEvalCap_list(result_list)
                    
                    try:
                        cocoEval_final.evaluate()
                    except:
                        print("Metric calc failed")

                    final_metrics = pd.DataFrame(cocoEval_final.final_metrics)
                    final_metrics = final_metrics.fillna(0).apply(lambda x: x.sum()).to_frame().T.apply(lambda x: x.round(2), axis=0)

                    sample_metrics = pd.concat([pd.DataFrame.from_dict(metric) for metric in cocoEval_final.sample_metrics], axis=1)


                    #%% get variances of softmaxed magic scores and model confidence

                    var_magic_scores = torch.tensor(var_magic_scores_list).cuda().unsqueeze(1)
                    var_model_conf = torch.tensor(var_model_conf_list).cuda().unsqueeze(1)
                    vars = torch.cat([var_magic_scores, var_model_conf], axis=-1)
                    

                    #%% create table and result .json                    

                    save_name_results_json = args.save_name
                    file_prefix = str(beta.item()) + "_" + prompt.replace(" ", "_") + "_" + "kappa" + "_" + str(one_res_dict["temperature"]) + "_" + save_name_results_json
                    html_filename =  file_prefix + "_" "results.html"
                    sim_audio_table = pd.concat(audio_sim_tables.values())
                    sim_audio_table = pd.concat([sample_metrics.reset_index(drop=True),\
                                                sim_audio_table.reset_index(drop=True)], axis=1)
                    sim_audio_table = pd.concat([final_metrics, sim_audio_table], axis=0)

   
                    if args.dataset == "clotho":
                        html_path = os.path.join(os.getcwd(), "../inference_result", args.language_model_name, "clotho_v2.1" , table_subfolder, "output_tables", args.experiment, html_filename)
                        result_jsons_full_save_path = os.path.join(os.getcwd(), "../inference_result", args.language_model_name, "clotho_v2.1", table_subfolder, "output_jsons", args.experiment, file_prefix + ".json")
                        print("Saving in Clotho results")

                    elif args.dataset == "audiocaps":
                        html_path = os.path.join(os.getcwd(), "../inference_result", args.language_model_name, "AudioCaps", table_subfolder, "output_tables", args.experiment, html_filename)
                        result_jsons_full_save_path = os.path.join(os.getcwd(), "../inference_result", args.language_model_name, "AudioCaps", table_subfolder, "output_jsons", args.experiment, file_prefix + ".json")
                        print("Saving in AudioCaps results")

                    else:
                        pass

                    print ('HTML_path: {}'.format(html_path))
                    print ('Results json path: {}'.format(result_jsons_full_save_path))

                    if os.path.exists(os.path.dirname(html_path)):
                        pass
                    else: # recursively construct directory
                        os.makedirs(os.path.dirname(html_path), exist_ok=True)

                    sim_audio_table.to_html(html_path, escape=False)
                
                    #print ('Inference completed!')

                    #print ('full save path is {}'.format(full_save_path))

                    if os.path.exists(os.path.dirname(result_jsons_full_save_path)):
                        pass
                    else: # recursively construct directory
                        os.makedirs(os.path.dirname(result_jsons_full_save_path), exist_ok=True)
                    
                    with open(result_jsons_full_save_path, 'w') as outfile:
                        json.dump(result_list, outfile, indent=4)

                    #torch.save(vars, os.path.join(os.getcwd(), "../inference_result/AudioCaps", table_subfolder, "output_jsons", 'variances_magic_model_conf.pt'))
