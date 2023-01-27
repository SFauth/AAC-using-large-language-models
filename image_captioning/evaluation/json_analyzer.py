#%% load dependencies

import json 
import os
import glob
import pandas as pd

#%% load json

path_to_json = os.path.join(os.getcwd(), "..", "inference_result/clotho_v2.1/magic")
json_pattern = os.path.join(path_to_json,'*.json')
file_list = glob.glob(json_pattern)
file_list_an_audio_clip_of_a = [file for file in file_list if "an_audio_clip_of_a" in file]


#%% column names

col_names = [file.split("v2.1_", 1)[1] for file in file_list_an_audio_clip_of_a]


#%%

res_dfs = []

for file_number, file in enumerate(file_list_an_audio_clip_of_a):

    with open(file) as json_file:
        res_dict = json.load(json_file)
        [dictionary.pop("split") for dictionary in res_dict]
        res_df = pd.DataFrame(res_dict).set_index("sound_name")
        res_df["captions"] = res_df["captions"].map(lambda x: x[0:2])
        res_df[col_names[file_number]] = res_df.values.tolist()
        res_df = res_df[~res_df.index.duplicated(keep='first')]
        res_dfs.append(res_df[[col_names[file_number]]])

res = pd.concat(res_dfs, axis=1)

pd.options.display.max_colwidth = 500 
pd.set_option('display.expand_frame_repr', True)

res.reset_index(drop=True)[["gpt2_partial_training_run_an_audio_clip_of_a_result_k_250_alpha_5.1_beta_20.json", \
    "run_an_audio_clip_of_a_result_k_250.json",\
        "gpt2_partial_training_run_an_audio_clip_of_a_result_k_250_alpha_5.1_beta_10.json"]].head(15)

# %%
