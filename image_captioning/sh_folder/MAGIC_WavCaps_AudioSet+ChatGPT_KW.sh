CUDA_VISIBLE_DEVICES="7" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name facebook/opt-1.3b\
    --audio_code_path ../clip/\
    --audio_pt_file ../clip/WavCaps/retrieval/assets/HTSAT-BERT-FT-AudioCaps.pt\
    --inference_file_prefix ../softlinks/AudioCaps_data/\
    --GT_captions_path ../data/AudioCaps/AudioCaps_code_exp.json\
    --decoding_len 78\
    --sample_rate 32000\
    --k 45\
    --save_name MAGIC_WavCaps_AudioSet+ChatGPT_KW\
    --dataset audiocaps\
    --include_prompt_magic False\
    --experiment test_performance\
    --path_to_AudioSet_keywords ../data/AudioSet/class_labels_indices.csv\
    --path_to_ChatGPT_keywords ../data/sounding_objects/chatgpt_audio_tags.csv &&
    
    python "../evaluation/join_test_results.py"\
    --result_files_path ../inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/evaluation/test_performance/\
    --hyperparam_json_path ../inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/output_jsons/test_performance/