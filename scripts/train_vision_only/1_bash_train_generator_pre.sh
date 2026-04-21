python -m train.generator.train_generator_pre \
--vq-model VQ-8 \
--gpt-model GPT-314M \
--epochs 20 \
--global-batch-size 32 \
--log-every 100 \
--ckpt-every 20000 \
--val-every 10000 \
--num-val-samples 64 \
--n-ref 8 \
--results-dir ./results/results_pre \
--data-dir-path ./data/fontimg \
--data-style-info-json ./data/split_style_info.json \
--data-content-info-json ./data/split_content_info.json \
--vq-ckpt /path/to/vq/tokenizer/checkpoint \
"$@"