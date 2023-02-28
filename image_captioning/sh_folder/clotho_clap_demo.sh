CUDA_VISIBLE_DEVICES=[6,7] python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name gpt2\
    --clap_path ../clip/\
    --clap_model_name ../clip/CLAP/assets/checkpoints/no_fusion_keyword_to_caption.pt\
    --test_image_prefix_path ../softlinks/audio_clip_test_data/\
    --test_path ../softlinks/test_json_cherry\
    --decoding_len 30\
    --sample_rate 48000\
    --k 45\
    --alpha 0.1\
    --beta 2.0\
    --save_path_prefix ../inference_result/clotho_v2.1/includes_prompt_magic/output_jsons/code_testing\
    --save_name magic_clotho_v2.1_clap_demo.json