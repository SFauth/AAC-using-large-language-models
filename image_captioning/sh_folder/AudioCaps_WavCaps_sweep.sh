CUDA_VISIBLE_DEVICES="3" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name facebook/opt-1.3b\
    --clap_path ../clip/WavCaps/retrieval/\
    --clap_model_name ../clip/WavCaps/retrieval/assets/HTSAT-BERT-FT-AudioCaps.pt\
    --test_image_prefix_path ../softlinks/AudioCaps_data/\
    --test_path ../data/AudioCaps/AudioCaps_code_exp.json\
    --decoding_len 16\
    --sample_rate 32000\
    --k 45\
    --alpha 0.1\
    --beta 2.0\
    --save_name wav_caps_demo\
    --dataset audiocaps\
    --include_prompt_magic False\
    --experiment hyperparhyperparam_experiment