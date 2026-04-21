python -m reward.style_reward_train \
--data-dir-path ./data/fontimg \
--data-style-info-json ./data/split_style_info.json \
--data-content-info-json ./data/split_content_info.json \
--image-size 64 \
--results-dir ./results/results_style_reward_model \
--epochs 10 \
--global-batch-size 1024 \
--global-seed 42 \
--log-every 100 \
--ckpt-every 2000 \
--val-every 1000 \
"$@"