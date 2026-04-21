#!/bin/bash
python -m sample.generator.sample_generator \
  --output-dir ./results_samples/vision_only/font6 \
  --vq-model VQ-8 \
  --gpt-model GPT-314M \
  --generator-ckpt /path/to/generator/checkpoint \
  --batch-size 128 \
  --all-char-list 啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍 \
  --generate-char-list 艾碍爱隘鞍 \
  --content-ref-dir-path ./data/sample/content_ref/font0 \
  --style-ref-dir-path ./data/sample/style_ref/font6 \
  --n-ref 8 \
  --ft-epoch 0 \
  --ft-layer-num 2 \
  --sample
