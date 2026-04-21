python -m language.extract_t5_features \
--t5-model-path /path/to/t5/model \
--max-token-len 128 \
--style-prompt-json-file-path ./data/font_description.json \
--t5-feat-save-path ./data/styles_t5embeddings \
"$@"
