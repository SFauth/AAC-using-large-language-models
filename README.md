## Using large pre-trained language models for audio captioning
**Author**: Stefan Fauth


This repository contains code, models, and other related resources of my master's thesis. The repo is based on the MAGIC paper [[Language Models Can See:
Plugging Visual Controls in Text Generation]](https://arxiv.org/abs/2205.02655). I am grateful to the authors: Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier.

****

![AACLM](/demo.png)

****
## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#environment_setup'>2. Environment setup</a>
* <a href='#data'>3. Data setup</a>
* <a href='#clip_models'>4. Audio CLIP Model setup</a>
* <a href='#stanford'>5. Natural language generation metrics setup </a>
* <a href='#run_inference'>6. Running inference</a>
* <a href='#future_work'>7. Future work</a>
    * <a href='#future_work_hyper'>7.1. How to exchange the hyperparameters that are not in the shell script?</a>
    * <a href='#future_work_LM'>7.2. How to exchange the language model? </a>
    * <a href='#future_work_CLIP'>7.3. How to exchange the audio CLIP model?</a>
    * <a href='#future_work_flag'>7.4. Detailed explanation of all flags in the inference shell-scripts</a>
* <a href='#contact'>8. Contact</a>
****

<span id='introduction'/>

### 1. Introduction:
Automated Audio Captioning using LMs.
****
<span id='environment_setup'/>

### 2. Environment setup:

1. Clone the repo
```
git clone https://github.com/SFauth/AACLM.git
```
2. Set it as the current directory
```
cd AACLM
```
3. Create a new conda environment with the correct Python version
```
conda create --name <ENV_NAME> python=3.7.1
```
4. Activate the environment
```
conda activate <ENV_NAME>
```
5. Install all dependencies
```
pip3 install -r requirements.txt
conda install -c conda-forge openjdk # for using Java based Stanford NLG metrics computation
```
****
<span id='data'/>

### 3. Data setup:

The repo is constructed, such that the data to do inference on can be stored in another directory. In case, you already have AudioCaps and Clotho's evaluation data set loaded, it is enough to **specify the softlink** in the directory audio_captioning/softlinks and you can skip a) and b). If you do **not have it** yet and want to minimize the effort to run an experiment, just **follow a) and b)**

- a) AudioCaps:
```
cd audio_captioning/softlinks
tar -xzf AudioCaps_data.tar.gz # unpack the compressed file containg the data
```
- b) Clotho :
Download the evaluation, i.e. test data, of Clotho Version 2.1:
```
wget https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z
mv clotho_audio_evaluation.7z evaluation_data_files.7z
7z x evaluation_data_files.7z
```
****

### 4. Audio CLIP Model setup:

<span id='clip_models'/>

Set up the pre-trained audio CLIP model's checkpoint. If you only want to use the best audio CLIP model, skip a) and b):

- a) For AudioCLIP:
```
cd audio_captioning/clip/AudioCLIP/assets
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
```
- b) For LAION:
```
cd audio_captioning/clip/CLAP/assets
wget https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
```
- c) For WavCaps:
   - Replace the file name with your absolute path to the inference.yaml file https://github.com/SFauth/AACLM/blob/62e2c0a29c1e6a9efc4f7e4e7becf40104df7465/audio_captioning/clip/load_clip_model.py#L8 and 
   - Download checkpoint:
```
cd audio_captioning/clip/WavCaps/retrieval/assets
gdown 1il6X1EiUPlbyysM9hn2CYr-YRSCuSy2m
```
****

<span id='stanford'/>

### 5. Natural language generation metrics setup:

```
bash audio_captioning/evaluation/get_stanford_models.sh
```

****
<span id='run_inference'/>

### 6. Running inference:

In the folder ``` audio_captioning/sh_folder ``` , there are two types of shell scripts. 
- Type A: _search_audioCLIPmodel_keywords.sh_ (inference scripts)
- Type B: _create_X.sh_ (visualization and table creating scripts)

In order to do inference: run the desired **Type A** script:
- MAGIC: MAGIC search on or greedy search with no guiding (off)
- Audio Model: which audio CLIP model to use
- Keywords: which keyword list to use for the Socratic prompt improvement
- Script: name of the shell script that has to be run to use the row's main components

**TLDR**: run the shell script MAGIC_WavCaps_AudioSet_KW.sh for the best model

**Disclaimer**: before running, specify in the chosen shell script, which GPU to use, e.g.:

```
CUDA_VISIBLE_DEVICES="1" 
```

|MAGIC| Audio Model | Keywords | Script | Comment |
|----------|----------|---------|--------|---------|
|Off| - | - | MAGIC_no_audio.sh | Baseline |
|Off| WavCaps | AudioSetKW | WavCaps_AudioSet_KW.sh |  |
|Off| WavCaps | AudioSet+ChatGPT KW | WavCaps_AudioSet+ChatGPT_KW.sh |  |
|On | AudioCLIP | - | MAGIC_AudioCLIP.sh | |
|On | AudioCLIP | AudioSetKW | MAGIC_AudioCLIP_AudioSet_KW.sh | |
|On | AudioCLIP | AudioSetKW+ChatGPT KW | MAGIC_AudioCLIP_AudioSet+ChatGPT_KW.sh | |
|On | LAION | - | MAGIC_LAION.sh | |
|On | LAION | AudioSetKW | MAGIC_LAION_Audioset_KW.sh | |
|On | LAION | AudioSetKW+ChatGPT KW | MAGIC_LAION_AudioSet+ChatGPT_KW.sh | |
|On | WavCaps | - | MAGIC_WavCaps.sh | |
|On | WavCaps | AudioSetKW | MAGIC_WavCaps_AudioSet_KW.sh | Best Model|
|On | WavCaps | AudioSetKW+ChatGPT KW | MAGIC_WavCaps_AudioSet+ChatGPT_KW.sh | |


**Inspecting the results**

There are three files to analyze the results of a run with a dataset, which are stored in the folder specifying the language model. A run is uniquely identified by its time suffix.

1. evaluation: A table containing the NLG metrics of the run: https://github.com/SFauth/AACLM/blob/1a9aa00c3af548f997a0aa6474ed31f0ed3ad303/audio_captioning/inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/evaluation/test_performance/0.193_2023-05-31%2009%3A01%3A59_MAGIC_WavCaps_AudioSet_KW.csv

2. output_tables: An HTML table containing the audio clip and sample-level results (NLG metrics, cosine similarities with the audio of the prediction, the prediction, ...). Run the HTML file
https://github.com/SFauth/AACLM/blob/1a9aa00c3af548f997a0aa6474ed31f0ed3ad303/audio_captioning/inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/output_tables/test_performance/0.193_2023-05-31%2009%3A01%3A59_MAGIC_WavCaps_AudioSet_KW.html

3. output_jsons: A list of dictionaries containing the prediction for every sample and all hyperparameters
https://github.com/SFauth/AACLM/blob/1a9aa00c3af548f997a0aa6474ed31f0ed3ad303/audio_captioning/inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/output_jsons/test_performance/0.193_2023-05-31%2009%3A01%3A59_MAGIC_WavCaps_AudioSet_KW.json

****
<span id='future_work'/>

### 7. Future work

How to exchange components of the system?

#### 7.1 How to **exchange** the **hyperparameters** that are **not in** the **shell-script**. 

<span id='future_work_hyper'/>

Change the parameter, e.g. $\beta$ and $l$:

- https://github.com/SFauth/AACLM/blob/d55733c5e74e67e4845c79f6616faf883b7b2069/audio_captioning/inference_magic.py#L218
- https://github.com/SFauth/AACLM/blob/d55733c5e74e67e4845c79f6616faf883b7b2069/audio_captioning/inference_magic.py#L229
  
   
#### 7.2 How to **exchange** the **language model** with another model from HuggingFace?

<span id='future_work_LM'/>

Note that this may vary, depending on the model.
- change in the corresponding shell-script the flag _language_model_name_ according to the model name on Huggingface, e.g. GPT2
- if necessary, adapt the snippet:
[https://github.com/SFauth/AACLM/blob/557f433fe7f7b369df23f156113b15cfa6b670ca/audio_captioning/language_model/simctg.py#L48-L52](https://github.com/SFauth/AACLM/blob/557f433fe7f7b369df23f156113b15cfa6b670ca/audio_captioning/language_model/simctg.py#L48-L52)

#### 7.3 How to **exchange** the **audio CLIP** model? 

<span id='future_work_CLIP'/>

Replicate the code for the other audio CLIP models
- add a preprocessing function to: https://github.com/SFauth/AACLM/blob/62e2c0a29c1e6a9efc4f7e4e7becf40104df7465/audio_captioning/clip/audio_preprocessors.py#L26-L29
- create a model loading function: https://github.com/SFauth/AACLM/blob/62e2c0a29c1e6a9efc4f7e4e7becf40104df7465/audio_captioning/clip/load_clip_model.py#L23-L35
- add an elif condition for the new model: https://github.com/SFauth/AACLM/blob/62e2c0a29c1e6a9efc4f7e4e7becf40104df7465/audio_captioning/inference_magic.py#L126-L135
- specify in the corresponding shell-script the path to the model's checkpoint: https://github.com/SFauth/AACLM/blob/62e2c0a29c1e6a9efc4f7e4e7becf40104df7465/audio_captioning/sh_folder/MAGIC_WavCaps_AudioSet_KW.sh#L5

#### 7.4 **Explanation** of every **flag**:

<span id='future_work_flag'/>

https://github.com/SFauth/AACLM/blob/62e2c0a29c1e6a9efc4f7e4e7becf40104df7465/audio_captioning/inference_magic.py#L32-L57

****

<span id='contact'/>

### 8. Contact
If there are still open questions, have a look at the dissertation or contact me at (SFauth@gmx.net).


****


