# coding=utf-8
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import argparse
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
from datetime import datetime
from re import sub, search
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from evaluation.pycocoevalcap.eval import COCOEvalCap_obs
from evaluation.pycocoevalcap.eval import COCOEvalCap_list

logging.getLogger('transformers.generation_utils').disabled = True

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # language model path setting
    parser.add_argument("--language_model_code_path", type=str, help="where is the code of language model located (folder)")
    # model and data configuration
    parser.add_argument("--language_model_name", type=str, help="HuggingFace name of pre-trained language model: gpt2 or facebook/opt-1.3b")
    # index path setting
    parser.add_argument("--audio_code_path", type=str, help="path to audio model's code (folder)")
    parser.add_argument("--audio_pt_file", type=str, help="path to .pt file of audio model")
    parser.add_argument("--inference_file_prefix", type=str, help="folder storing files to do inference on (inference_file_prefix/audioclip_0.wav)")
    # test data path
    parser.add_argument("--GT_captions_path", type=str, help="JSON file connecting GT captions and audio file names")
    # decoding configuration
    parser.add_argument("--decoding_len", type=int, default=16, help='maximum length of (prefix + generated continuation)')
    # sample rate 
    parser.add_argument("--sample_rate", type=int, default=44100)
    # magic configuration
    parser.add_argument("--k", type=int, default=-1, help='k for magic search')
    # save configuration
    parser.add_argument("--save_name", type=str, help="save name suffix of output json containing hyperparams, GT captions and prediction")
    parser.add_argument("--experiment", type=str, help="specify: hyperparam_experiments or code_testing")
    # data set
    parser.add_argument("--dataset", type=str, help="specify dataset: clotho or AudioCaps")
    # prompt
    parser.add_argument("--include_prompt_magic", type=str, help="include prompt in the calculation of the MAGIC score")
    # keywords
    parser.add_argument("--path_to_AudioSet_keywords", type=str, help="creates an intermediate prompt using AudioSet keywords")
    parser.add_argument("--path_to_ChatGPT_keywords", type=str, help="creates an intermediate prompt using AudioSet+ChatGPT keywords")

    # latex table
    parser.add_argument("--latex_table_description", type=str, help="creates a LaTeX table containing NLG \
                        metrics and the specified table description (what has been done to get the table)")


    return parser.parse_args()


def get_prompt_id(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").input_ids   # gets token id's for the text: e.g. hello I am cool [50257, 281, 6597, 10651, 286, 257]
    return tokens




import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
        print('{} GPUs available'.format(torch.cuda.device_count()))
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    print ('Loading data...')

    with open(args.GT_captions_path) as f:
        item_list = json.load(f)
    print ('Data loaded.')

    print ('Number of test instances is {}'.format(len(item_list)))
    
    # get AudioCLIP

    sys.path.append(args.audio_code_path)  # define path to clap class

    seed = 4182
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if "WavCaps" in args.audio_pt_file:
        print('WavCaps encoder selected!')
        sys.path.append(os.path.join(args.audio_code_path, 'WavCaps/retrieval'))
        from load_clip_model import load_WavCaps
        from audio_preprocessors import preprocess_for_WavCaps
        clip = load_WavCaps(cuda_available,
                            device,
                            pt_file=args.audio_pt_file)
        
        preprocessor = preprocess_for_WavCaps
        
        
    elif "audioclip" in args.audio_pt_file:
        print('AudioCLIP encoder selected!')
        sys.path.append(os.path.join(args.audio_code_path, 'AudioCLIP'))
        from load_clip_model import load_AudioClip
        from audio_preprocessors import preprocess_for_AudioCLIP
        clip = load_AudioClip(cuda_available,
                            device,
                            pt_file=args.audio_pt_file)
        
        preprocessor = preprocess_for_AudioCLIP


    elif "CLAP" in args.audio_pt_file:
        print('CLAP encoder selected!')
        sys.path.append(os.path.join(args.audio_code_path, 'CLAP'))
        from load_clip_model import load_CLAP
        from audio_preprocessors import preprocess_for_CLAP
        clip = load_CLAP(cuda_available,
                            device,
                            pt_file=args.audio_pt_file)
        
        preprocessor = preprocess_for_CLAP

    print ('Loading off-the-shelf language model...')
    import sys
    sys.path.append(args.language_model_code_path)
    from simctg import SimCTG

    generation_model = SimCTG(args.language_model_name)

    if cuda_available:
        generation_model = generation_model.to(device)
    #accelerator.prepare_model(generation_model)
    generation_model.eval()  # set model to evaluation (test mode, i.e. no training)

    print ('Language model loaded.')
    clip_text_max_len = 77

    # prepare keywords

    if args.path_to_AudioSet_keywords != None and args.path_to_ChatGPT_keywords == None:


        print('Using AudioSet file as keyword list')
        keywords = list(pd.read_csv(args.path_to_AudioSet_keywords)["display_name"])

        # Create one entry for every tag (split "Male speech, man speaking" to "Male speech", "man speaking")

        keywords = [tag.strip() for tag in keywords for tag in tag.split(',')]

        with torch.no_grad():
            keyword_embeds = clip.module.encode_text(keywords)

    elif "chatgpt" in args.path_to_ChatGPT_keywords:

        print('Using AudioSet+ChatGPT files as keyword list')
        keywords_raw = list(pd.read_csv(args.path_to_AudioSet_keywords)["display_name"])
        keywords_raw = [tag.strip() for tag in keywords_raw for tag in tag.split(',')]
        chatgpt_keywords = pd.read_csv(args.path_to_ChatGPT_keywords, header=None)[0].tolist()
        keywords = list(set(keywords_raw + chatgpt_keywords))
        
        with torch.no_grad():

            def batch_list(input_list, batch_size):
                return [input_list[index:index+batch_size] for index in range(0, len(input_list), batch_size)]

            batched_keywords = batch_list(keywords,
                                          batch_size=1000)

            if "CLAP" in args.audio_pt_file:

                keyword_embeds = torch.cat([clip.encode_text(keyword_batch, use_tensor=True) for keyword_batch in batched_keywords])    

            else:
                keyword_embeds = torch.cat([clip.encode_text(keyword_batch) for keyword_batch in batched_keywords])


    else:
        print('No keywords used! ')
    

    betas = torch.linspace(0.1, 2, steps=2).cuda()


    prompts = ["This is a sound of "] 

    #prompts = [" "]


    temperatures = torch.linspace(10, 25, steps=2).cuda()

    top_keywords = torch.tensor([10]).cuda()

    #keywords_prompts = ["I am an intelligent audio captioning bot. I think there might be "]
     
    keywords_prompts = ["I hear "]

    #keywords_prompts = ["Generate an audio caption based on the objects "]
    
    include_prompt_magic = [False]

    alphas = torch.linspace(0, 0.1, steps=3).cuda()

    end_penaltys = torch.linspace(0.1, 0.16, steps=3).cuda()

    hyperparam_grid = itertools.product(betas,
                                        prompts,
                                        temperatures,
                                        top_keywords,
                                        keywords_prompts,
                                        include_prompt_magic,
                                        alphas,
                                        end_penaltys)
    

    for hyperparam in hyperparam_grid:
        
        beta = hyperparam[0]
        prompt = hyperparam[1]
        clip.logit_scale_a = hyperparam[2].unsqueeze(dim=0)
        l = hyperparam[3]
        keyword_prompt = hyperparam[4]
        include_prompt_magic = hyperparam[5]
        alpha = hyperparam[6]
        end_penalty = hyperparam[7]

        """
        beta = torch.linspace(0.5, 0.5, steps=1).cuda()
        clip.logit_scale_a = torch.linspace(18.6612, 18.6612, steps=1).cuda().unsqueeze(dim=0)
        l = torch.tensor([1]).cuda()
        keyword_prompt = "We heard "
        include_prompt_magic = True
        alpha = torch.linspace(0.1, 0.1, steps=1).cuda()
        """

        print("Beta: " + str(np.round(beta.item(),1)) \
              + " , Prompt: "  + prompt \
                + " , Temperature: " + str(np.round(clip.logit_scale_a.item(),1)) \
                    + ", l_keywords: " + str(l.item()) \
                        + ", keyword_prompt: " + str(keyword_prompt) \
                            +", include_prompt_in_MAGIC_search: " + str(include_prompt_magic) \
                                +", Alpha: " + str(np.round(alpha.item(),1)) \
                                    +", end_penalty : " + str(np.round(end_penalty.item(),5)))

        result_list = []
        invalid_num = 0
        print ('----------------------------------------------------------------')
        with torch.no_grad():
            test_num = len(item_list)
            #test_num = 10
            print ('Number of inference instances is {}'.format(test_num))
            
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
                
                sound_full_path = args.inference_file_prefix + one_test_dict['sound_name']

                # create sound instance 
                try:
                    sound_instance, _ = librosa.load(sound_full_path, sr=args.sample_rate, mono=True)

                    sound_instance = preprocessor(sound_instance).type(torch.cuda.FloatTensor)
                    
                    if "CLAP" in str(type(clip)):
                        audio_embeds = clip.encode_audio(sound_instance,
                                                        use_tensor=True)      
                    else:
                        audio_embeds = clip.encode_audio(sound_instance)
                    
                                            
                    if ((args.path_to_AudioSet_keywords != None) and (l != 0)):
                        
                        keyword_sounds_sims = torch.cosine_similarity(audio_embeds, keyword_embeds)
                        top_l_indices = torch.topk(keyword_sounds_sims, k=l).indices
                        top_l_objects = [keywords[index] for index in top_l_indices]
                        last_key_index = len(top_l_objects) - 1

                        if top_l_objects[last_key_index][0] in 'aeiouAEIOU':
                            last_object_with_article = "an " + top_l_objects[last_key_index]

                        else: 
                            last_object_with_article = "a " + top_l_objects[last_key_index]

                        top_l_objects_one_string = "{0}, and {1}".format(", ".join(top_l_objects[:last_key_index]), last_object_with_article)
                        
                        temp_prompt = keyword_prompt + top_l_objects_one_string +  " in this audio clip. " + prompt
                    
                        # tokenize prompt
                        input_ids = get_prompt_id(temp_prompt, generation_model.tokenizer)

                    else:

                        input_ids = get_prompt_id(prompt, generation_model.tokenizer)
                    
                    """
                    input ids: vector containing tokens of prompt [50257, 271, 6597, 10651, 286, 257]
                    """

                    if cuda_available:
                        input_ids = input_ids.to(device)

                    #try:

                    """
                    input_ids: gets token id of the SOS token 
                    """

                    output_text = generation_model.magic_search(input_ids, args.k, alpha, args.decoding_len, 
                        beta, audio_embeds, clip, clip_text_max_len, include_prompt_magic, end_penalty)
                    
                    # keep all which is preceeded by prompt
                    #res = search(prompt + r"\s+(.*)", output_text)
                    # output_text_without_prompt = res.group(1)
                    output_text_without_prompt = output_text.split(prompt)[-1]                
                    output_text_series = pd.Series(output_text)
                    output_text_without_prompt_series = pd.Series(output_text_without_prompt)
                    
                    one_res_dict['prediction'] = output_text_without_prompt # always without prompt, as prompt is other entry
                    one_res_dict["beta"] = beta.item()
                    one_res_dict["prompt"] = prompt
                    one_res_dict["k"] = args.k
                    one_res_dict["alpha"] = alpha.item()
                    one_res_dict["decoding_len"] = args.decoding_len
                    one_res_dict["clip_text_max_len"] = clip_text_max_len
                    one_res_dict["n_test_samples"] = test_num
                    one_res_dict["included_prompt_in_magic"] = include_prompt_magic
                    one_res_dict["dataset"] = args.dataset
                    one_res_dict["CLAP_type"] = os.path.split(args.audio_pt_file)[-1]
                    one_res_dict["temperature"] = clip.logit_scale_a.item()
                    one_res_dict["l"] = l.item()
                    one_res_dict["keyword_prompt"] = keyword_prompt
                    one_res_dict["end_penalty"] = end_penalty.item()

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
                    
                    if include_prompt_magic == True:
                        captions.append(output_text_without_prompt)
                        pred = output_text_without_prompt_series
                        table_subfolder = "includes_prompt_magic"
                    
                    else: 
                        captions.append(output_text_without_prompt)
                        pred = output_text_without_prompt_series
                        table_subfolder = "excludes_prompt_magic"


                    #%% 5) cosine similarity of the [GT captions_i] and the [prediction] (all with each other; matrix)
                    # includes prompt if specified in flag
                    
                    if "AudioCLIP" in str(type(clip)):
                        captions_as_lists = [[caption] for caption in captions]
                        captions_embs = clip.encode_text(captions_as_lists)

                    elif "CLAP" in str(type(clip)): 
                        captions_embs = clip.encode_text(captions, use_tensor=True).to('cuda' if torch.cuda.is_available() else 'cpu') 

                    else:
                        captions_embs = clip.encode_text(captions)

                    cos_sim_captions_list = cosine_similarity(captions_embs.cpu().detach().numpy()).round(2).astype(str).tolist()
                    [row.append('<br>') for row in cos_sim_captions_list]
                    cos_sim_captions_list = [val for sublist in cos_sim_captions_list for val in sublist]
                    cos_sim_captions = pd.Series({"cos_sim_captions":cos_sim_captions_list})
                    cos_sim_captions = pd.DataFrame(cos_sim_captions)[0].apply(' '.join)


                    #%% 3) 4) cosine similarity of the [GT_caption_i] and the [audio] and [prediction] and the [audio]
                    
                    cos_sim = torch.cosine_similarity(audio_embeds, captions_embs)# unscaled!                   
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

                    if "facebook" in args.language_model_name:
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
        mean_metrics = final_metrics.mean(axis=1).rename("Mean_NLG_M").round(decimals=3)
        final_metrics = pd.concat([mean_metrics, final_metrics], axis=1)

        sample_metrics = pd.concat([pd.DataFrame.from_dict(metric) for metric in cocoEval_final.sample_metrics], axis=1)
        

        #%% create table and result .json                    

        save_name_results_json = args.save_name

        file_prefix = str(np.round(mean_metrics.item(),3)) + \
                                        "_" + \
                                           datetime.today().strftime('%Y-%m-%d %H:%M:%S') + \
                                                "_" + \
                                                    save_name_results_json

        html_filename =  file_prefix + ".html"
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

        #if args.latex_table_description != None:

        #    sim_audio_table.to_latex
    
        #print ('Inference completed!')

        #print ('full save path is {}'.format(full_save_path))

        if os.path.exists(os.path.dirname(result_jsons_full_save_path)):
            pass
        else: # recursively construct directory
            os.makedirs(os.path.dirname(result_jsons_full_save_path), exist_ok=True)
        
        with open(result_jsons_full_save_path, 'w') as outfile:
            json.dump(result_list, outfile, indent=4)

        #torch.save(vars, os.path.join(os.getcwd(), "../inference_result/AudioCaps", table_subfolder, "output_jsons", 'variances_magic_model_conf.pt'))
