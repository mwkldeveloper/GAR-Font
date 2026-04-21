python -m train.tokenizer.train_tokenizer \
--vq-model VQ-8 \
--epochs 20 \
--global-batch-size 16 \
--log-every 100 \
--ckpt-every 20000 \
--val-every 10000 \
--num-val-samples 64 \
--codebook-weight 1.0 \
--reconstruction-weight 1.0 \
--perceptual-weight 0.001 \
--lr 1e-4 \
--data-dir-path ./data/fontimg \
--data-style-info-json ./data/split_style_info.json \
--data-content-info-json ./data/split_content_info.json \
"$@"