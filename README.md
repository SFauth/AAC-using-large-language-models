## Using large pre-trained language models for audio captionining
**Author**: Stefan Fauth


This repository contains code, models, and other related resources of my master's thesis. It is based on the MAGIC paper [[Language Models Can See:
Plugging Visual Controls in Text Generation]](https://arxiv.org/abs/2205.02655). I am grateful to the authors: Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier.

****

![MAGIC](/demo.gif)

****
## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#environment_setup'>2. Environment Setup</a>
* <a href='#data'>3. Loading AudioCaps and Clotho data</a>
* <a href='#clip_models'>4. Download audio CLIP models</a>
    * <a href='#image_captioning_experiment'>5.1. Implementation of Experiments</a>
    * <a href='#image_captioning_magic_search'>5.2. Example Usage of Magic Search</a> [![Open In Colab](https://colab.research.google.com/assets/colab-
****

<span id='introduction'/>

### 1. Introduction:
Generative language models (LMs) such as GPT-2/3 can be prompted to generate text with remarkable quality. While they are designed for text-prompted generation, it remains an open question how the generation process could be guided by modalities beyond text such as images. In this work, we propose a training-free framework, called MAGIC (i<ins>**MA**</ins>ge-<ins>**G**</ins>uided text generat<ins>**I**</ins>on with <ins>**C**</ins>LIP), for plugging in visual controls in the generation process and enabling LMs to perform multimodal tasks (e.g., image captioning) in a zero-shot manner. MAGIC is a simple yet efficient plug-and-play framework, which directly combines an off-the-shelf LM (i.e., GPT-2) and an image-text matching model (i.e., CLIP) for image-grounded text generation. During decoding, MAGIC influences the generation of the LM by introducing a CLIP-induced score, called **_magic score_**, which regularizes the generated result to be semantically related to a given image while being coherent to the previously generated context. Notably, the proposed decoding scheme does not involve any gradient update operation, therefore being computationally efficient. On the challenging task of zero-shot image captioning, MAGIC outperforms the state-of-the-art method by notable margins with a nearly 27 times decoding speedup. MAGIC is a flexible framework and is theoretically compatible with any text generation tasks that incorporate image grounding. In the experiments, we showcase that it is also capable of performing visually grounded story generation given both an image and a text prompt.
****
<span id='environment_setup'/>

### 2. Environment Setup:

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
```
****
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
```
cd audio_captioning/clip/WavCaps/retrieval/assets
gdown 1il6X1EiUPlbyysM9hn2CYr-YRSCuSy2m
```

****
### 5. Running inference:
In the folder ``` audio_captioning/sh_folder ``` , there are two types of shell scripts. 
- Type A: _create_X.sh_ 
- Type B: _search_audioCLIPmodel_keywords.sh_  all other remaining scripts

In order to do inference: **ignore Type A**


|MAGIC | Audio Model | Keywords | Script |
|----------|----------|---------|--------|
|Off| - | - | MAGIC_no_audio.sh |


   |Off |           - &                   - &    18.8 &     1.1 &     0.0 &     0.0 &     4.1 &     17.8 &    0.1 &    0.0 &     0.0 \\
   On &   AudioCLIP &                   - &    18.8 &     1.1 &     0.0 &     0.0 &     4.1 &     17.8 &    0.1 &    0.0 &     0.0 \\
   On &   AudioCLIP & AudioSet+ChatGPT KW &    26.0 &     6.7 &     0.7 &     0.0 &     7.6 &     21.3 &    1.2 &    2.0 &     1.6 \\
   On &   AudioCLIP &         AudioSet KW &    33.6 &    11.4 &     3.9 &     1.4 &     7.2 &     23.5 &    2.5 &    1.1 &     1.8 \\
   On &       LAION &                   - &    21.7 &     5.6 &     2.1 &     0.0 &     5.2 &     20.0 &    1.6 &    1.7 &     1.7 \\
   On &       LAION & AudioSet+ChatGPT KW &    41.1 &    21.5 &    11.2 &     5.2 &    11.7 &     30.4 &   26.1 &    7.8 &    17.0 \\
   On &       LAION &         AudioSet KW &    43.5 &    23.1 &    11.9 &     5.4 &    12.0 &     31.9 &   26.3 &    8.0 &    17.2 \\
   On &     WavCaps &                   - &    22.1 &     6.7 &     2.7 &     0.0 &     5.6 &     20.5 &    2.3 &    2.1 &     2.2 \\
   On &     WavCaps & AudioSet+ChatGPT KW &    39.5 &    20.0 &    10.4 &     4.8 &    10.9 &     29.2 &   20.7 &    7.0 &    13.8 \\
   On &     WavCaps &         AudioSet KW &    44.5 &    25.1 &    14.0 &     6.7 &    12.3 &     33.2 &   29.1 &    8.5 &    18.8 \\
  Off &     WavCaps & AudioSet+ChatGPT KW &    39.2 &    19.8 &    10.4 &     4.8 &    10.7 &     29.0 &   20.2 &    6.8 &    13.5 \\
  Off &     WavCaps &         AudioSet KW &    43.9 &    24.6 &    13.5 &     6.3 &    12.0 &     32.8 &   28.1 &    8.2 &    18.1 \\



For every model version, there is a shell script. For instance, for the best model this is the file: MAGIC_WavCaps_AudioSet_KW.sh.

What do you have to change in order to run the experiments? Remember that the current directory is the sh_folder (specify the paths in a way that goes out of this folder)

1. Specify which GPU to use:
```
CUDA_VISIBLE_DEVICES="1" 
```



<span id='data'/>


****

<span id='clip_models'/>

### 4. Prepare audio CLIP models:

It is structured as follows:
```
CUDA_VISIBLE_DEVICES="1" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name facebook/opt-1.3b\     # name of the language model on HuggingFace
    --audio_code_path ../clip/\
    --audio_pt_file ../clip/WavCaps/retrieval/assets/HTSAT-BERT-PT.pt\      # path to the pre-trained audio CLIP model's checkpoint
    --AudioCaps_inference_file_prefix ../softlinks/AudioCaps_data/\     # specify directory containing AudioCaps data 
    --clotho_inference_file_prefix ../softlinks/evaluation_data_files/\       # specify directory containing Clotho data 
    --GT_captions_AudioCaps ../data/AudioCaps/AudioCaps_test.json\       # replace with ../data/AudioCaps/AudioCaps_test.json if you want to use the validation set
    --GT_captions_clotho ../data/Clotho/clotho_v2.1_test.json\
    --decoding_len 78\
    --sample_rate 32000\
    --k 45\
    --save_name MAGIC_WavCaps_AudioSet_KW\
    --include_prompt_magic False\
    --experiment test_performance\
    --path_to_AudioSet_keywords ../data/AudioSet/class_labels_indices.csv     # specify path to keyword list
```

#### 5.1. Implementation of Experiments: 
To ensure the reproductity of our work, we provide all related resources to implement our experiments on the task of zero-shot image captioning. Please refer more details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning). 

<span id='image_captioning_magic_search'/>

#### 5.2. Example Usage of Magic Search: 
In the following, we illustrate how to perform zero-shot image captioning with magic search. Specifically, we show how to generate the results as shown in our case study in the paper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NDVkKpanbsaUwecHoRp_2kIpMztOFW25?usp=sharing)

<span id='image_captioning_language_model'/>

<span id='image_captioning_experiment'/>
- Go to https://github.com/cdjkim/audiocaps/tree/master/dataset, download val.csv and test.csv and put them into data/AudioCaps/
- Run process_AudioCaps.py to prepare the data 
##### 5.2.1. Load Language Model:
We first load the language model as:
```python
import sys
sys.path.append(r'./image_captioning/language_model/')
from simctg import SimCTG
language_model_name = r'cambridgeltl/magic_mscoco'
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
generation_model = SimCTG(language_model_name, sos_token, pad_token)
generation_model.eval()
```

<span id='image_captioning_CLIP'/>

##### 5.2.2. Load CLIP: 
Then, we load the CLIP model as:
```python
import sys
sys.path.append(r'./image_captioning/clip/')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()
```

<span id='image_captioning_start_token'/>

##### 5.2.3. Prepare Start Token: 
Note that, the language model always starts generation with a start of sentence token. Here, we prepare the input ids of the start token.
```python
import torch
sos_token = r'<-start_of_text->'
start_token = generation_model.tokenizer.tokenize(sos_token)
start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
input_ids = torch.LongTensor(start_token_id).view(1,-1)
```

<span id='image_captioning_load_image'/>

##### 5.2.4. Load Image: 
To generate the caption of a random image, we need to load the image as:
```python
from PIL import Image             # to load images
from IPython.display import display # to display images
image_name_list = ['COCO_val2014_000000336777.jpg', 'COCO_val2014_000000182784.jpg', 'COCO_val2014_000000299319.jpg', 'COCO_val2014_000000516750.jpg',
                   'COCO_val2014_000000207151.jpg', 'COCO_val2014_000000078707.jpg', 'COCO_val2014_000000027440.jpg', 'COCO_val2014_000000033645.jpg',
                   'COCO_val2014_000000348905.jpg', 'COCO_val2014_000000545385.jpg', 'COCO_val2014_000000210032.jpg', 'COCO_val2014_000000577526.jpg']
index = 1 
'''
   you can easily reproduce all results shown in our case study (index from 0 to 3) 
   and the results in the appendix (index from 4 to 11).
'''

image_path = r'./image_captioning/example_images/' + image_name_list[index]
image_instance = Image.open(image_path)
display(image_instance)
```

<img src="https://github.com/yxuansu/MAGIC/blob/main/image_captioning/example_images/COCO_val2014_000000182784.jpg" width="400" height="280">


<span id='image_captioning_magic_search_result'/>

##### 5.2.5. Zero-Shot Image Captioning with Magic Search: 
Now, let's generate the image caption with magic search!
```python
'''
   setup the configurations of magic search
      k: the k in magic search
      alpha: the alpha in magic search
      beta: the beta in magic search
      decoding_len: the number of tokens to generate
'''
k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
eos_token = '<|endoftext|>'
output = generation_model.magic_search(input_ids, k, 
        alpha, decoding_len, beta, image_instance, clip, 60)
print (output)
'''
   A large cow standing in a street stall.
'''
```

<span id='image_captioning_reproduce_result'/>

##### 5.2.6. Reproduce Our Results in the Paper: 
If you would like to reproduce all the results shown in the case study and appendix of our paper, you can run this demo [file](https://github.com/yxuansu/MAGIC/blob/main/image_caption_demo.py) as

```yaml
python image_caption_demo.py
```

****

<span id='story_generation'/>

### 6. Visually Grounded Story Generation:

<span id='story_generation_experiment'/>

#### 6.1. Implementation of Experiments: 
To ensure the reproductity of our work, we provide all related resources to implement our experiments on the task of visually grounded story generation. Please refer more details [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation). 

<span id='story_generation_magic_search'/>

#### 6.2. Example Usage of Magic Search: 
In the following, we illustrate how to perform visually grounded story generation with magic search. Specifically, we show how to generate the results as shown in our case study in the paper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19lyyMXDRNr-Op8vwUOiRmbhMxI_s3rwW?usp=sharing)

<span id='story_generation_language_model'/>

##### 6.2.1. Load Language Model: 
We first load the language model and prepare the story title as:
```python
import sys
sys.path.append(r'./story_generation/language_model')
from transformers import AutoTokenizer
from simctg import SimCTG
language_model_name = r'cambridgeltl/simctg_rocstories'
tokenizer = AutoTokenizer.from_pretrained(language_model_name)
generation_model = SimCTG(language_model_name, tokenizer.pad_token_id)
generation_model.eval()

import torch
title = 'Ice Cream Tasting <|endoftext|>'
title_tokens = tokenizer.tokenize(title)
title_id_list = tokenizer.convert_tokens_to_ids(title_tokens)
title_ids = torch.LongTensor(title_id_list).view(1,-1)
```

<span id='story_generation_CLIP'/>

##### 6.2.2. Load CLIP: 

Then, we load the CLIP model as:
```python
import sys
sys.path.append(r'./story_generation/clip')
from clip import CLIP
model_name = "openai/clip-vit-base-patch32"
clip = CLIP(model_name)
clip.eval()
```

<span id='story_generation_get_image'/>

##### 6.3.2. Get the Related Image: 
Next, let's get the images that are related to the story tile. We provide **two** ways of doing it as shown below:

<span id='story_generation_get_image_from_index'/>

###### 6.3.2.1. Retrieve from Image Index: 
The first way is to retrieve the images from a constructed image index. Before running the following commands, please make sure you have built the image index from scrath as described [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/image_index#1-build-image-index-from-scratch) or downloaded our provided image index as described [[here]](https://github.com/yxuansu/MAGIC/tree/main/story_generation/data#1-prepare-image-index).

After the image index is ready, we can load the image index as
```python
# build image index
import sys
sys.path.append(r'./story_generation/image_index')
from imageindex import ImageIndex
index_path = r'./story_generation/data/image_index/images_index_data/index_matrix.txt'
mapping_dict_path = r'./story_generation/data/image_index/images_index_data/mapping_dict.json'
image_folder_prefix_path = r'./story_generation/data/image_index/images/'
index = ImageIndex(index_path, mapping_dict_path, image_folder_prefix_path, clip)
```

Then, we can retrieve the top-1 images as
```python
image_name_list, image_instance_list = index.search_image(title, top_k=1)
'''
   image_name_list: the list of names of the retrieved images
   image_instance_list: the list of images that we retrieve
'''
```

Let's see the retrieved images we got
```python
from IPython.display import display
# display the top-1 image
display(image_instance_list[0])
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/avopix-284658167.jpg" width="360" height="280">


<span id='story_generation_get_image_from_example'/>

###### 6.3.2.2. Directly Load Image: 
Alternatively, if you have not prepared the image index, we have provided these the image in the repo. You can directly load it as
```python
from PIL import Image
image_name_list = ['avopix-284658167.jpg']
image_instance_list = []
for name in image_name_list:
    image_path = r'./story_generation/example_images/' + name
    image_instance = Image.open(image_path)
    image_instance_list.append(image_instance)
```

<span id='story_generation_magic_search_result'/>

##### 6.3.3. Visually Grounded Story Generation with Magic Search: 
**[Note]** Recall that, in this example, our story title is 'Ice Cream Tasting <|endoftext|>'.

Now, let's generate the story conditioned on the retrieved image
```python
from IPython.display import display
k, alpha, beta, decoding_len  = 5, 0.6, 0.15, 100
'''
   The k, alpha, beta correspond to the k, alpha, beta in magic search
'''
image_instance = image_instance_list[0]
eos_token = r'<|endoftext|>'
output, _ = generation_model.magic_search(title_ids, k, alpha, decoding_len, beta, image_instance, 
        clip, 60, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
display(image_instance)
'''
   My family went to a ice cream shop. They ordered three flavors of ice cream. The first one was 
   strawberry, the second was chocolate, and the third was orange. I was excited to try all three 
   flavors. It was very good and I had a great time at the ice cream shop.
'''
```
<img src="https://github.com/yxuansu/MAGIC/blob/main/story_generation/example_images/avopix-284658167.jpg" width="360" height="280">

Then, let's see what we can get using the vanilla contrastive search **without** the image grounding.
```python
k, alpha, decoding_len  = 5, 0.6, 100
'''
   The k and alpha correspond to the k and alpha in contrastive search
'''
eos_token = r'<|endoftext|>'
output, _ = generation_model.fast_contrastive_search(title_ids, k, alpha, decoding_len, eos_token)
_, generated_story = generation_model.parse_generated_result(output, num_of_sentences_to_keep=5)
print (generated_story)
'''
   My family went to a ice cream shop. We ordered the Ice Cream Truck. It was delicious. The customer 
   service was terrible. We had to leave for another day.
'''
```

<span id='story_generation_reproduce_result'/>

##### 6.3.4. Reproduce Our Results in the Paper: 
If you would like to reproduce all the results shown in the case study and appendix of our paper, you can run this demo [file](https://github.com/yxuansu/MAGIC/blob/main/story_generation_demo.py) as

```yaml
python story_generation_demo.py
```


****

<span id='contact'/>

### 7. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).


****

<span id='magic_elsewhere'/>

### 8. MAGIC Elsewhere
We thank the community's effort for extending MAGIC!

- [Replicate](https://replicate.com/home) has provided a great [[demo]](https://replicate.com/yxuansu/magic/examples) of MAGIC that is super easy to use. Thanks for the effort!

