CUDA_VISIBLE_DEVICES="1" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name gpt2\
    --clap_path ../clip/WavCaps/retrieval/\
    --clap_model_name ../clip/WavCaps/retrieval/assets/HTSAT-BERT-FT-AudioCaps.pt\
    --test_image_prefix_path ../softlinks/AudioCaps_data/\
    --test_path ../data/AudioCaps/AudioCaps_val.json\
    --decoding_len 16\
    --sample_rate 32000\
    --k 45\
    --alpha 0.1\
    --beta 2.0\
    --save_name sweep_incl_temperature_no_chat_gpt\
    --dataset audiocaps\
    --include_prompt_magic False\
    --experiment hyperparam_experiments