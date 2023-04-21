CUDA_VISIBLE_DEVICES="4" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name facebook/opt-1.3b\
    --clap_path ../clip/WavCaps/retrieval/\
    --clap_model_name ../clip/WavCaps/retrieval/assets/HTSAT-BERT-FT-AudioCaps.pt\
    --test_image_prefix_path ../softlinks/AudioCaps_data/\
    --test_path ../data/AudioCaps/AudioCaps_val.json\
    --decoding_len 40\
    --sample_rate 32000\
    --k 45\
    --alpha 0.1\
    --beta 2.0\
    --save_name ext_keywords\
    --dataset audiocaps\
    --include_prompt_magic False\
    --experiment hyperparam_experiments\
    --path_to_AudioSet_keywords ../data/AudioSet/class_labels_indices.csv\
    --path_to_ChatGPT_keywords ../data/sounding_objects/chatgpt_audio_tags.csv