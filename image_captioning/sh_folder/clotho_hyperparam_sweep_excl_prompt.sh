CUDA_VISIBLE_DEVICES="1" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name gpt2\
    --clap_path ../clip/\
    --clap_model_name ../clip/CLAP/assets/checkpoints/no_fusion_no_keyword_to_caption.pt\
    --test_image_prefix_path ../softlinks/evaluation_data_files/\
    --test_path ../softlinks/full_test_data_json\
    --decoding_len 30\
    --sample_rate 48000\
    --k 45\
    --alpha 0.1\
    --beta 2.0\
    --save_name exp\
    --dataset clotho\
    --include_prompt_magic False\
    --experiment hyperparam_experiments