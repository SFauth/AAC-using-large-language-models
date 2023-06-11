## Using large pre-trained language models for audio captioning
**Author**: Stefan Fauth


This repository contains code, models, and other related resources of my master's thesis. The repo is built on MAGIC: we have built on MAGIC's code related to MAGIC search, the repo structure and the sh-folders to build our system. I am grateful to the authors: Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Please check out their [repository](https://github.com/yxuansu/MAGIC) and [paper](https://arxiv.org/abs/2205.02655). 

Except for this README, there are different README files in the subfolders. However, if you only want to do inference using the best model, you can ignore these and focus on the main README here.


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
* <a href='#validation_docs'>9. Validation documentation: what hyperparam grid did we use</a>
* <a href='#contact'>10. Contact</a>
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
6. Secret Tip

If you have problems with the environment, just activate the MAGIC environment in the Callisto directory in my AACLM directory and run everything there.
****
<span id='data'/>

### 3. Data setup:

The shell scripts are written in a way that requires the data to be in

```
audio_captioning/softlinks
```

The structure should look like:

      .
      ├── ./softlinks/  
         ├── AudioCaps_data  # stores AudioCaps .wav files
             ├── audiocaps_audio_clip_1.wav
             ├── ...
             └── ...
         ├── evaluation_data_files # stores Clotho .wav files
             ├── clotho_audio_clip_1.wav
             ├── ...
             └── ...


If you want to store the data in a different folder, **specify the softlink** or change in **every shell script** the flag:
https://github.com/ExplainableML/2023-audiocaptioning-msc-stefan/blob/e3c8b02345b8999e1a274a86e1f1804498e3788d/audio_captioning/sh_folder/MAGIC_AudioCLIP.sh#L6-L7



In case, you already have AudioCaps and Clotho's evaluation data set loaded, it is enough to **specify the softlink** in the directory and you can skip a) and b). 

If you do **not have the data** yet, just **follow a) and b)**:


- a) AudioCaps (Download validation and test data):

```
cd audio_captioning/softlinks
mkdir AudioCaps_data
cd AudioCaps_data
apt install ffmpeg youtube-dl
aac-datasets-download --root "." audiocaps --subsets "val"
aac-datasets-download --root "." audiocaps --subsets "test"
```
Put all .wav files into ``` AudioCaps_data ```.
You can use ```mv AUDIOCAPS_32000Hz/audio/test/*.wav .``` and ```mv AUDIOCAPS_32000Hz/audio/val/*.wav .```

Make sure that the file names match with the file names in ```audio_captioning/data/AudioCaps/AudioCaps_val.json```, resp.
```audio_captioning/data/AudioCaps/AudioCaps_test.json```. If this is not the case, you can either change all file names, such that they match.

Alternatively, check out the script ```audio_captioning/data/process_AudioCaps.py``` and create new .JSON files that match them with their GT captions.



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
1. evaluation: Stores a .csv file containing the NLG metrics of every run:

|Dataset| Model | Mean NLG Score | ... | SPIDEr |
|----------|----------|---------|--------|---------|
|AudioSet| MAGIC_WavCaps_AudioSet_KW_l_test_ablation | 0.166 | ... | 0.138 |

2. output_tables: Stores an HTML table containing the audio clip and sample-level results for qualitative analysis (NLG metrics, cosine similarities with the audio of the prediction, the prediction, ...). Run an HTML file to view it!

3. output_jsons: For every run a list of dictionaries containing the prediction for every sample and all hyperparameters:


```yaml
{
        "split": "test",
        "sound_name": "--0w1YA1Hm4_30000.wav",
        "captions": [
            "A vehicle driving as a man and woman are talking and laughing",
            "Men speak and laugh with humming of an engine",
            "High pitched speaking and laughing",
            "Humming of an engine with a woman and men speaking",
            "People talking with the dull roar of a vehicle on the road"
        ],
        "prediction": "laughter and joy.",
        "beta": 0.5,
        "prompt": "This is a sound of ",
        "k": 45,
        "alpha": 0,
        "decoding_len": 78,
        "clip_text_max_len": 77,
        "n_test_samples": 975,
        "included_prompt_in_magic": false,
        "dataset": "AudioCaps",
        "CLAP_type": "HTSAT-BERT-PT.pt",
        "temperature": 10,
        "l": 7,
        "keyword_prompt": "Objects: ",
        "end_penalty": 0.10000000149011612
    }
```

After deciding on which result file (CSV, HTML or JSON) you want to check out, specify the experiment type:
- validation = runs on the validation set to find the optimal hyperparameters (validation plot: Figure 2)
- test_performance = runs on the test set for model ablation (results in the state-of-the-art table (SOTA) and in the model ablation plot: Table 6, 7 and Figure 3)
- ablation = runs on the test set for hyperparamer ablation (values in the $\beta$ and $l$ ablation plots made with the best model found in validation: Figure 4, 5)

A run is uniquely identified by its time suffix. Like this, the three files can be matched for every run. The figure in front of the timestamp is the average of all NLG metrics, indicating the quality of the run.

**TLDR**: There are **three files** per run on a dataset. Click through all subfolders and pick the ones with the most recent timestamp to check on your results.

If this was not clear, check out the ```audio_captioning/README.md```, which sketches the folder structure.

If you want to check out my specific logs, go on Callisto in my home directory to:

```
code/AACLM/audio_captioning/inference_results
```

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
<span id='validation_docs'/>

### 9. Validation documentation

We provide a small summary of the hyperparameter grids that we ran full sweeps on with the best model to find the optimal set of hyperparameters:


|Dataset| Hyperparameter | Values |
|----------|----------|----------|
|AudioCaps Validation| $\alpha$ | [0, 0.5] | 
|...| $\beta$ |[0, 0.1, 0.2, 0.3, ... 2] | 
|...| $\gamma$ |[0.1, 0.13, 0.16]|
|...| $\tau$ |[10, 18.7, 25]|
|...| $l$ | [0, 1, 2] |
|...| $k$ | [45] |
|...| keyword_prompt | [Objects: ] |
|...| basic_prompt | [This is a sound of ] |


We have then used this set doing our main component ablation studies and have also ablated $\beta$ and $l$.






****

<span id='contact'/>

### 10. Contact
If there are still open questions, have a look at the dissertation or contact me at (SFauth@gmx.net).


****


