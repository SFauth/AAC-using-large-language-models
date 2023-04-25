CUDA_VISIBLE_DEVICES="5" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name facebook/opt-1.3b\
    --audio_code_path ../clip/\
    --audio_pt_file ../clip/AudioCLIP/assets/audioclip_pt\
    --AudioCaps_inference_file_prefix ../softlinks/AudioCaps_data/\
    --clotho_inference_file_prefix ../softlinks/evaluation_data_files/\
    --GT_captions_AudioCaps ../data/AudioCaps/AudioCaps_test.json\
    --GT_captions_clotho ../data/Clotho/clotho_v2.1_test.json\
    --decoding_len 78\
    --sample_rate 44100\
    --k 45\
    --save_name MAGIC_AudioCLIP\
    --include_prompt_magic False\
    --experiment test_performance &&
    
    python "../evaluation/join_test_results.py"\
    --result_files_path ../inference_result/facebook/opt-1.3b/\
    --hyperparam_json_path ../inference_result/facebook/opt-1.3b/AudioCaps/excludes_prompt_magic/output_jsons/test_performance/