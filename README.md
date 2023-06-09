## Using large pre-trained language models for audio captioning
**Author**: Stefan Fauth


This repository contains code, models, and other related resources of my master's thesis. The repo is based on MAGIC. I am grateful to the authors: Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Please check out their [repository](https://github.com/yxuansu/MAGIC) and [paper](https://arxiv.org/abs/2205.02655). 



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
* <a href='#logs'>7. Inspecting results / logs </a>
* <a href='#future_work'>8. Future work</a>
    * <a href='#future_work_hyper'>8.1. How to exchange the hyperparameters that are not in the shell script?</a>
    * <a href='#future_work_LM'>8.2. How to exchange the language model? </a>
    * <a href='#future_work_CLIP'>8.3. How to exchange the audio CLIP model?</a>
    * <a href='#future_work_flag'>8.4. Detailed explanation of all flags in the inference shell-scripts</a>
    * <a href='#ablation'>8.5. Example: how did we do the ablation studies to find the optimal beta?</a>
* <a href='#contact'>9. Contact</a>
****

<span id='introduction'/>

### 1. Introduction:
Automated audio captioning (AAC) is concerned with obtaining a meaningful
caption of an audio clip. Usually, this problem is approached by expensive training
of an encoder and a decoder. This training process requires significant computa-
tional capacities and is data-hungry. In the field of AAC, data scarcity is still a major
problem and the current datasets are significantly smaller than datasets in image cap-
tioning. Completely avoiding any training, we propose, to our knowledge, the first
zero-shot AAC system that, also, only consists of off-the-shelf pre-trained compo-
nents. We build a system, based on recent advances in computer vision (CV), where
the combination of pre-trained components have achieved impressive performance
on image captioning problems in a zero-shot setting. By using a pre-trained audio
CLIP model, we identify meaningful keywords that describe the audio clip and create
an audio-guided prompt. We then use this prompt to make use of the capabilities
of a large pre-trained language model (LM) and guide its decoding process. Instead
of making the model greedily choose the token featuring the highest probability,
we aurally guide the selection of the next token using the same audio CLIP model.
We show that our model is able to clearly outperform a baseline on AudioCaps and
Clotho, using common metrics in AAC research. In order to explain these results,
we conduct experiments with three different audio CLIP models that differ in pre-
training data and architecture. Moreover, we provide extensive ablation studies with
different hyperparameters that control the amount of guiding in the prompt and in
the decoding process. Finally, we use the insights from these results to provide clear
perspectives for future research, which can be easily adapted due to the flexibility of
the framework.
****
<span id='environment_setup'/>

### 2. Environment setup:

1. Clone the repo
```
git clone https://github.com/SFauth/AACLM.git](https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan.git
```
2. Set it as the current directory
```
cd 2023-audiocaptioning-msc-stefan

```
3. Create a new conda environment with the correct Python version
```
conda env create -f env1.yaml
```
4. Activate the environment
```
conda activate ZSAAC
```
5. Install remaining dependencies
```
conda install -c conda-forge openjdk # for using Java based Stanford NLG metrics computation
```
****
<span id='data'/>

### 3. Data setup:

In case, you already have AudioCaps and Clotho's evaluation data set loaded, it is enough to **specify the softlink** in the directory audio_captioning/softlinks and you can skip a) and b). If you do **not have it** yet, just **follow a) and b)**:

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

Set up the pre-trained audio CLIP model's checkpoint. If you want to directly reproduce our best model, skip a) and b):

- a) For AudioCLIP:
   AudioCLIP also requires a vocabulary file for the tokenizer (2nd link).
```
cd audio_captioning/clip/AudioCLIP/assets
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
wget -P https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz
```
- b) For LAION:
```
cd audio_captioning/clip
mkdir CLAP/assets
cd CLAP/assets
wget https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
```
- c) For WavCaps:
   - Replace the file name with your absolute path to the inference.yaml file https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/clip/load_clip_model.py#L8 and 
   - Download the checkpoint:
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
|Off| - | - | baseline.sh | Baseline |
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

****
<span id='logs'/>

### 7. Inspecting results / logs:


The folder ```audio_captioning/inference_results/facebook/opt-1.3b``` stores the result of all runs. If a new language model is selected, the program should automatically generate a new folder for the new model. Inside the LM's folder, there is one folder for each dataset:
1. AudioCaps/excludes_prompt_magic
2. clotho_v2.1/excludes_prompt_magic

Inside each folder, there are three subfolders analyzing the results.
1. evaluation: Stores a .csv file containing the NLG metrics of every run: https://github.com/SFauth/AACLM/blob/1a9aa00c3af548f997a0aa6474ed31f0ed3ad303/audio_captioning/inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/evaluation/test_performance/0.193_2023-05-31%2009%3A01%3A59_MAGIC_WavCaps_AudioSet_KW.csv
2. output_tables: Stores an HTML table containing the audio clip and sample-level results for qualitative analysis (NLG metrics, cosine similarities with the audio of the prediction, the prediction, ...). Run an HTML file to view it!
https://github.com/SFauth/AACLM/blob/1a9aa00c3af548f997a0aa6474ed31f0ed3ad303/audio_captioning/inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/output_tables/test_performance/0.193_2023-05-31%2009%3A01%3A59_MAGIC_WavCaps_AudioSet_KW.html
3. output_jsons: For every run a list of dictionaries containing the prediction for every sample and all hyperparameters
https://github.com/SFauth/AACLM/blob/1a9aa00c3af548f997a0aa6474ed31f0ed3ad303/audio_captioning/inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/output_jsons/test_performance/0.193_2023-05-31%2009%3A01%3A59_MAGIC_WavCaps_AudioSet_KW.json

After deciding on which result file (CSV, HTML or JSON) you want to check out, specify the experiment type:
- validation = runs on the validation set to find the optimal hyperparameters
- test_performance = runs on the test set for model ablation (results in the state-of-the-art table (SOTA) and in the model ablation plot)
- ablation = runs on the test set for hyperparamer ablation (values in the $\beta$ and $l$ ablation plots made with the best model in test_performance)

A run is uniquely identified by its time suffix. Like this, the three files can be matched for every run. The figure in front of the timestamp is the average of all NLG metrics, indicating the quality of the run.

**TLDR**: There are **three files** per run on a dataset. Click through all subfolders and pick the ones with the most recent timestamp to check on your results.

If this was not clear, check out the ```audio_captioning/README.md```, which sketches the folder structure.

****
<span id='future_work'/>

### 8. Future work

How to exchange components of the system or conduct experiments? We have found the optimal parameters by running sweeps. This can be done by making a change and then using one of the shell-scripts. 

#### 8.1 How to **exchange** the **hyperparameters** that are **not in** the **shell-script**?

<span id='future_work_hyper'/>

Change the parameter, e.g. $\beta$ and $l$:

- https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/inference_magic.py#L216
- https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/inference_magic.py#L225
  
   
#### 8.2 How to **exchange** the **language model** with another model from HuggingFace?

<span id='future_work_LM'/>

Note that this may vary, depending on the model.
- change in the corresponding shell-script the flag _language_model_name_ according to the model name on Huggingface, e.g. GPT2
- if necessary, adapt the snippet:
https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/language_model/simctg.py#L48-L52

#### 8.3 How to **exchange** the **audio CLIP** model? 

<span id='future_work_CLIP'/>

Replicate the code for the other audio CLIP models
- add a preprocessing function to: https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/clip/audio_preprocessors.py#L26-L29
- create a model loading function: https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/clip/load_clip_model.py#L23-L25
- add an elif condition for the new model: https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/inference_magic.py#L124-L133
- specify in the corresponding shell-script the path to the model's checkpoint: https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/sh_folder/MAGIC_AudioCLIP_AudioSet_KW.sh#L5

#### 8.4 **Explanation** of every **flag**:

<span id='future_work_flag'/>

https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/inference_magic.py#L31-L60

#### 8.5 Example: how to do validation runs? 

<span id='ablation'/>

1. According to <a href='#future_work_hyper'> 7.1 How to exchange the hyperparameters that are not in the shell-script </a>, we have exchanged $\beta$ with the values we wanted to try out:
   ```python
   betas = torch.tensor([0, 0.1, 0.2, 0.3], device=device) 
   ```
Do the same for every value that you want to try out. 
   
2. Create a shell-script and define  a GPU to use. Since we want to use the validation set of AudioCaps, we have to **define the correct --GT_captions_AudioCaps flag** that contains the names of the files that are part of the validation set. Furthermore, we **specify as experiment name "validation"** to make the result being saved in the validation folder and **set a save name containing the hyperparameter that we ablate**: https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/sh_folder/MAGIC_WavCaps_AudioSet_KW_hyperparam_sweep_AC.sh#LL1C1-L14C1
   
3. As we have had than more than one GPU, we then repeated step 1 and step 2 using other $\beta$ or hyperparameter values, indicating to use another GPU
   
4. Use a jupyter notebook to create the visualization: https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/c4a8fcdc479ab3c77cb79f59d3f82cdba7d2c933/audio_captioning/evaluation/development_result_jsons/create_val_plot.ipynb
****

<span id='contact'/>

### 9. Contact
If there are still open questions, have a look at the dissertation or contact me at (SFauth@gmx.net).


****


