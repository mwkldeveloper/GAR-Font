python -m train.generator.train_generator_nfa \
--vq-model VQ-8 \
--gpt-model GPT-314M \
--epochs 50 \
--global-batch-size 32 \
--log-every 100 \
--ckpt-every 10000 \
--val-every 2000 \
--num-val-samples 128 \
--n-ref 8 \
--nfa-ori-train-num 16 \
--results-dir ./results/results_nfa \
--data-dir-path ./data/fontimg \
--data-style-info-json ./data/split_style_info.json \
--data-content-info-json ./data/split_content_info.json \
--data-nfa-dir-path ./data/nfa_folder \
--generator-ckpt /path/to/generator/checkpoint \
"$@"