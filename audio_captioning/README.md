## Zero-Shot Image Captioning
In this directory, we illustrate the details of our experiments on the task of zero-shot image captioning. 

> ****  The structure of the directory looks like:

    .
    ├──
        ├── ./data/  # Contains the keyword lists and preprocessing functions to create the JSONs combining file_name and corresponding GT captions
        ├── ./language_model/ # Stores the code of the language model
        ├── ./clip/  # Contains the code of the audio CLIP models. Their checkpoints can also be stored here
        ├── ./evaluation/ # Stores the code to create the plots and tables of the dissertation
        ├── ./sh_folder/ # Contains the shell-scripts to infer and to create the tables / plots
        ├── ./inference_result/ # Contains the inferenced results (one JSON, one CSV and one HTML per run) each run is uniquely identified by the timestamp
        └── sound_obj_generator.py # This program created the ChatGPT keyword list
        

**[Note]** To ensure the reproductivity of our work, [in this folder](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/inference_result), we provide the inferenced results of all evaluated methods.

****
### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#unsupervised_domain_adaptation'>2. Unsupervised Domain Adaptation</a>
    * <a href='#mscoco_adaptation'>2.1. MSCOCO</a>
    * <a href='#flickr30k_adaptation'>2.2. Flickr30k</a>
    * <a href='#huggingface_models'>2.3. Huggingface Models</a>
* <a href='#inference_with_magic'>3. Perform Inference with Magic</a>
* <a href='#inference_with_baseline'>4. Perform Inference with Baseline Methods</a>
    * <a href='#topk_sampling'>4.1. Top-k Sampling</a>
    * <a href='#nucleues_sampling'>4.2. Nucleus Sampling</a>
    * <a href='#contrastive_search'>4.3. Contrastive Search</a>
    * <a href='#clipre'>4.4. CLIPRe</a>
    * <a href='#zerocap'>4.5. ZeroCap</a>
* <a href='#evaluation'>5. Perform Evaluation</a> 


****

<span id='data_preparation'/>

### 1. Data Preparation:
To prepare the data for MSCOCO and Flickr30k benchmarks, please follow instructions [[here]](https://github.com/yxuansu/MAGIC/tree/main/image_captioning/data).


****

