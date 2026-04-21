python -m train.generator_adapter.train_generator_adapter \
--vq-model VQ-8 \
--gpt-model GPT-314M \
--epochs 20 \
--global-batch-size 128 \
--log-every 100 \
--ckpt-every 5000 \
--val-every 5000 \
--num-val-samples 32 \
--t5-feat-len 128 \
--train-num-per-font 1024 \
--results-dir ./results/results_generator_adapter_keep4 \
--n-ref 8 \
--adapter-n-keep 4 \
--data-dir-path ./data/fontimg \
--data-style-info-json ./data/split_style_info.json \
--data-content-info-json ./data/split_content_info.json \
--data-t5-feat-path ./data/styles_t5embeddings \
--generator-ckpt /path/to/generator/checkpoint \
"$@"