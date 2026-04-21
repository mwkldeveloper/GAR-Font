#!/bin/bash
python -m sample.generator_adapter.sample_generator_adapter \
  --output-dir ./results_samples/vision_language_adapter/font6 \
  --t5-model-path /data/caihn/AR_Font/TEXT_ALIGN_DATA/google \
  --vq-model VQ-8 \
  --gpt-model GPT-314M \
  --generator-ckpt /path/to/generator/checkpoint \
  --adapter-ckpt /path/to/adapter/checkpoint \
  --input-jsonl ./data/sample/adapter_sample_jsonl.jsonl \
  --batch-size 128 \
  --max-token-len 140 \
  --vision-n-ref 4 \
  --sample
  

