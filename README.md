## Language Models can See: Plugging Visual Controls in Text Generation


****
## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#news'>2. News</a>
* <a href='#citation'>3. Citation</a>
* <a href='#environment_setup'>4. Environment Setup</a>
* <a href='#image_captioning'>5. Zero-Shot Image Captioning</a>
    * <a href='#image_captioning_experiment'>5.1. Implementation of Experiments</a>
    * <a href='#image_captioning_magic_search'>5.2. Example Usage of Magic Search</a>
         * <a href='#image_captioning_language_model'>5.2.1. Load Language Model</a>
         * <a href='#image_captioning_CLIP'>5.2.2. Load CLIP</a>
         * <a href='#image_captioning_start_token'>5.2.3. Prepare Start Token</a>
         * <a href='#image_captioning_load_image'>5.2.4. Load Image</a>
         * <a href='#image_captioning_magic_search_result'>5.2.5. Zero-Shot Image Captioning with Magic Search</a>
* <a href='#story_generation'>6. Visually Grounded Story Generation</a>
    * <a href='#story_generation_experiment'>6.1. Implementation of Experiments</a>
    * <a href='#story_generation_magic_search'>6.2. Example Usage of Magic Search</a>
         * <a href='#story_generation_language_model'>6.2.1. Load Language Model</a>
* <a href='#contact'>7. Contact</a>

****

<span id='introduction'/>

### 1. Introduction:

****

<span id='news'/>

### 2. News:

****

<span id='citation'/>

### 3. Citation:
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@article{DBLP:journals/corr/abs-2202-06417,
  author    = {Yixuan Su and
               Tian Lan and
               Yan Wang and
               Dani Yogatama and
               Lingpeng Kong and
               Nigel Collier},
  title     = {A Contrastive Framework for Neural Text Generation},
  journal   = {CoRR},
  volume    = {abs/2202.06417},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.06417},
  eprinttype = {arXiv},
  eprint    = {2202.06417},
  timestamp = {Fri, 18 Feb 2022 12:23:53 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-06417.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

****

<span id='environment_setup'/>

### 4. Environment Setup:
```yaml
python version: 3.8
pip3 install -r requirements.txt
```

****

<span id='image_captioning'/>

### 5. Zero-Shot Image Captioning:

<span id='image_captioning_experiment'/>

#### 5.1. Implementation of Experiments: 
To ensure the reproductity of our work, we provide all related resources to implement our experiments on the task of zero-shot image captioning. Please refer more details [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning). 

<span id='image_captioning_magic_search'/>

#### 5.2. Example Usage of Magic Search: 
In the following, we illustrate how to perform zero-shot image captioning with magic search. Specifically, we show how to generate the results as shown in our case study in the paper.

<span id='image_captioning_language_model'/>

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

**[Note]** If you are looking for a python file that runs the above example, you can run this demo [file](https://github.com/yxuansu/MAGIC/blob/main/image_caption_demo.py) to see the results.

****

<span id='story_generation'/>

### 6. Visually Grounded Story Generation:

<span id='story_generation_experiment'/>

#### 6.1. Implementation of Experiments: 




****

<span id='contact'/>

### 7. Contact
If you have any questions, feel free to contact me via (ys484 at cam.ac.uk).

