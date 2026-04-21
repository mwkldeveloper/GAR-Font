<div align=center>

# Beyond Patches: Global-aware Autoregressive Model for Multimodal Few-Shot Font Generation

</div>

<p align="center">
  <img src="figures/teaser.jpg" width="80%"/>
</p>

<div align=center>

[![arXiv preprint](http://img.shields.io/badge/arXiv-2601.01593-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.01593) 
[![Homepage](https://img.shields.io/badge/Homepage-GAR--Font-orange)](https://xtryer-s.github.io/projects_pages/GAR_Font/)
[![Code](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/xTryer-s/GAR-Font)

</div>

---

# 📌 Overview

GAR-Font is designed for controllable font generation under limited style references.
It supports two major settings:

* **Vision-Only GAR-Font**
  Generate glyphs from a few reference glyph images.

* **Vision-Language GAR-Font**
  Generate glyphs using both visual references and natural language style descriptions.


# ⚙️ Environment Preparation

Create the conda environment and install dependencies:

```bash
conda create -n GAR_Font python=3.9 -y
conda activate GAR_Font

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install requests tqdm einops peft ftfy beautifulsoup4 easyocr
```

---

# 📂 Data Preparation

Examples could be found in ./data.
## 1. Fonts Folder

Collect `.ttf` fonts and render glyph images into folders:

```bash
Fonts_Folder/
├── font1/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── font2/
├── font3/
└── ...
```

Each image corresponds to one character index.

---

## 2. Content Info JSON

Defines all characters and train/test splits.

```json
{
  "all_content": [
    {
      "char": "啊",
      "index": 0
    },
    {
      "char": "唉",
      "index": 5
    }
  ],
  "train_content": [
    {
      "char": "啊",
      "index": 0
    }
  ],
  "test_content": [
    {
      "char": "唉",
      "index": 5
    }
  ]
}
```

* `all_content`: all characters
* `train_content`: characters used during training
* `test_content`: unseen characters for evaluation

---

## 3. Style Info JSON

Defines style splits.

```json
{
  "basic_style": [
    "basic_font_style1"
  ],
  "train_style": [
    "train_font_style1"
  ],
  "test_style": [
    "test_font_style1"
  ],
  "content_ref_style": [
    "content_ref_font_style"
  ]
}
```

* `basic_style`: used in Structural Enhancement Stage for unseen characters
* `train_style`: training fonts
* `test_style`: unseen fonts
* `content_ref_style`: font used as content reference

---

## 4. NFA Folder

Used in the NFA training stage.

```bash
NFA_Folder/
├── font1/
│   ├── nfa_glyph1.png
│   ├── nfa_glyph2.png
│   └── ...
├── font2/
└── ...
```

---

# 🏋️ Training GAR-Font

Model settings can be modified in:

```bash
./model/model_config.py
```

---

# 1️⃣ Vision-Only GAR-Font

## Step 1: Train G-Tok

```bash
bash ./scripts/train_vision_only/0_bash_train_tokenizer.sh \
--data-dir-path /path/to/fonts/folder \
--data-style-info-json /path/to/style/info/json \
--data-content-info-json /path/to/content/info/json
```

---

## Step 2: Pretrain GAR-Font

```bash
bash ./scripts/train_vision_only/1_bash_train_generator_pre.sh \
--data-dir-path /path/to/fonts/folder \
--data-style-info-json /path/to/style/info/json \
--data-content-info-json /path/to/content/info/json \
--vq-ckpt /path/to/trained/G-Tok/checkpoint \
--n-ref /num/of/vision/refs/for/Vision-Only-GAR-Font
```

---

## Step 3: NFA Stage

```bash
bash ./scripts/train_vision_only/2_bash_train_generator_nfa.sh \
--data-dir-path /path/to/fonts/folder \
--data-style-info-json /path/to/style/info/json \
--data-content-info-json /path/to/content/info/json \
--data-nfa-dir-path /path/to/nfa/folder \
--generator-ckpt /path/to/trained/GAR-Font/checkpoint
```

---

## Step 4: SE Stage

### 4.1 Train Style Discriminator

```bash
bash ./reward/bash_train_style_reward_model.sh
```

### 4.2 Structural Enhancement for GAR-Font

```bash
bash ./scripts/train_vision_only/3_bash_train_generator_se.sh \
--data-dir-path /path/to/fonts/folder \
--data-style-info-json /path/to/style/info/json \
--data-content-info-json /path/to/content/info/json \
--generator-ckpt /path/to/trained/GAR-Font/checkpoint \
--style-reward-model-ckpt /path/to/trained/StyleRewardModel/checkpoint
```

---

# 2️⃣ Vision-Language GAR-Font

---

## Step 1: Caption Fonts

Use a VLM (e.g. Qwen2.5-VL, SmolVLM2) to generate style descriptions for each font. You may also use real human-annotated text corpora with manually written font style descriptions.

Prepare a JSON file:

```json
{
  "FontA": "Style Description of FontA.",
  "FontB": "Style Description of FontB."
}
```

---

## Step 2: Extract T5 Embeddings

Download [flan-t5-xl](https://huggingface.co/google/flan-t5-xl) as the language tokenizer from Hugging Face. Then extract the t5 embeddings of font descriptions.


```bash
bash ./scripts/train_vision_langugae/0_bash_extract_t5_embedding.sh \
--t5-model-path /path/to/flan-t5/folder \
--style-prompt-json-file-path /path/to/font/caption/json \
--t5-feat-save-path /path/to/result/t5embedding/folder
```

---

## Step 3: Train Vision-Language Adapter

```bash
bash ./scripts/train_vision_langugae/1_bash_train_adapter.sh \
--data-dir-path /path/to/fonts/folder \
--data-style-info-json /path/to/style/info/json \
--data-content-info-json /path/to/content/info/json \
--data-t5-feat-path /path/to/t5embedding/folder \
--generator-ckpt /path/to/trained/GAR-Font/checkpoint \
--n-ref /num/of/vision/refs/to/be/aligned \
--adapter-n-keep /num/of/vision/refs/for/Multimodal-GAR-Font
```

---

# 🎨 Sampling With GAR-Font

---

# 1️⃣ Vision-Only Sampling

```bash
bash ./scripts/sample/bash_sample_vision_generator.sh \
--generator-ckpt /path/to/trained/GAR-Font/checkpoint \
--batch-size 128 \
--all-char-list 啊阿埃挨哎唉哀皑 \
--generate-char-list 啊阿埃挨 \
--content-ref-dir-path /path/to/Fonts_Folder/content_ref_font_style \
--style-ref-dir-path /path/to/style/ref/dir
```

* `all-char-list`: full character list for index lookup
* `generate-char-list`: target characters to generate
* `style-ref-dir-path`: organized similarly to `NFA_Folder`

---

# 2️⃣ Vision-Language Sampling

```bash
bash ./scripts/sample/bash_smaple_vision_language_generator_adapter.sh \
--generator-ckpt /path/to/trained/GAR-Font/checkpoint \
--adapter-ckpt /path/to/trained/adapter/checkpoint \
--input-jsonl /path/to/sample/jsonl
```

---

## Example JSONL Input

```json
{
  "content_image": "/path/to/content_ref.png",
  "style_images": [
    "/path/to/style_ref1.png",
    "/path/to/style_ref2.png"
  ],
  "style_prompt": "A font style that .."
}
```

* `content_image`: reference glyph for content
* `style_images`: few-shot style references
* `style_prompt`: natural language style description

---


## 📖 Citation

If you find this project useful, please cite our paper.

```bibtex
@misc{cai2026patchesglobalawareautoregressivemodel,
      title={Beyond Patches: Global-aware Autoregressive Model for Multimodal Few-Shot Font Generation}, 
      author={Haonan Cai and Yuxuan Luo and Zhouhui Lian},
      year={2026},
      eprint={2601.01593},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.01593}, 
}
```

---

## 🙏 Acknowledgement

Our implementation is based on [LlamaGen](https://github.com/FoundationVision/LlamaGen).
