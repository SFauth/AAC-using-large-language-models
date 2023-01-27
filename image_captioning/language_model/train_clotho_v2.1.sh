CUDA_VISIBLE_DEVICES=6 python train.py\
    --model_name gpt2\
    --train_path ../data/clotho_v2.1/clotho_v2.1_train.json\
    --dev_path ../data/clotho_v2.1/clotho_v2.1_val.json\
    --test_path ../data/clotho_v2.1/clotho_v2.1_test.json\
    --add_eos_token_to_data True\
    --margin 0.5\
    --max_len 64\
    --number_of_gpu 1\
    --batch_size_per_gpu 32\
    --gradient_accumulation_steps 4\
    --effective_batch_size 128\
    --total_steps 20000\
    --print_every 100\
    --save_every 500\
    --learning_rate 2e-5\
    --save_path_prefix ./unsupervised_clotho_v2.1/