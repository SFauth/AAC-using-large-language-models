CUDA_VISIBLE_DEVICES=6 python ../inference_baseline.py\
    --language_model_code_path ../language_model/magic_mscoco\
    --language_model_name cambridgeltl/magic_mscoco\
    --test_path ../data/clotho_v2.1/clotho_v2.1_test.json\
    --decoding_method topk\
    --decoding_len 16\
    --top_k 40\
    --save_path_prefix ../inference_result/clotho_v2.1/baselines/\
    --save_name top_k_result_run_2.json
