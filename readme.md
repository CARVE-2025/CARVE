# CARVE: Region-aware Identification Framework for Learning Exploits  
Official implementation & reproduction guide
============================================

CARVE detects software vulnerabilities by combining AST-level region
partition, region-aware contrastive pre-training, and risk-gated
classification.  
This repository contains all code and data needed to reproduce the
results in our paper.

---

## Dataset
* FFMPeg+Qemu: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
* BigVul: <https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing>
* DiverseVul: <https://drive.google.com/drive/folders/1BeX33sgLOWLBnJ_vjcYitzz87F1kFZWi?usp=drive_link>
> **After download & extraction**
> 1. **Split the code** — ensure every compilation unit is stored as an individual  
>    `*.c` file.  
> 2. Move these files into `dataset/<Corpus>/c/`.  

## 1 Project structure

```

CARVE/
├── joern/                     # Joern CLI (binary) + `extract-ast.sc`
├── dataset/
│   ├── FFMPeg+Qemu/
│   │   ├── c/                 # raw .c files
│   │   ├── js/                # AST JSONs (after Step 3a)
│   │   ├── W2V/               # word2vec model
│   │   └── vulnerable_lines.json
│   ├── BigVul/                # same layout + cwe_mapping.json
│   └── DiverseVul/            # same layout + cwe_mapping.json
├── CWE/                       # per-CWE PKLs (generated in Step 3b)
│   ├── BigVul/
│   └── DiverseVul/
├── main.py                    # train / evaluate
├── model.py                   # network definition
├── training.py                # training loop
├── data_preprocessing.py      # dataset-level preprocessing
├── data_preprocessing-CWE.py  # CWE-level preprocessing
└── data_loading.py

````

---

## 2 Prerequisites

| Item           | Tested version         |
|----------------|------------------------|
| Python         | 3.8.2                  |
| PyTorch + CUDA | 2.4.1 / 12.1           |
| Joern CLI      | 2.0.86                 |
| Others         | see `requirements.txt` |

```bash
cd CARVE
pip install -r requirements.txt
````

---

## 3 Data preparation

### 3a  Generate AST JSONs with Joern

```bash
# (1) Add Joern CLI to PATH
export PATH="$PATH:/your-path/CARVE/joern/joern-cli"

# (2) Run extraction script
joern --script joern/extract-ast.sc
```

This scans every `.c` file under `dataset/*/c/` and writes a matching
`*.json` AST to `dataset/<name>/js/`.

### 3b  Dataset-level preprocessing (function scope)

```bash
python data_preprocessing.py --dataset "FFMPeg+Qemu"
python data_preprocessing.py --dataset "BigVul"
python data_preprocessing.py --dataset "DiverseVul"
```

Each call produces `<dataset>.pkl` beside the raw folders.

### 3c  CWE-level preprocessing (BigVul / DiverseVul)

```bash
python data_preprocessing-CWE.py --dataset "BigVul"
python data_preprocessing-CWE.py --dataset "DiverseVul"
```

Creates one `.pkl` per CWE under `CWE/<dataset>/`.

---

## 4 Training & evaluation

### 4a  Function-level end-to-end training

```bash
python main.py \
  --gpus 0 \
  --data_path "FFMPeg+Qemu.pkl"
```

`main.py` first performs **region-aware contrastive pre-training** then
automatically launches **risk-gated classification**, finally printing
test metrics.

### 4b  CWE-level evaluation

1. **Train unified model**

   ```bash
   python main.py \
     --gpus 0 \
     --cwe_mode \
     --data_path "CWE/BigVul/"
   ```

   Checkpoints:

   * `clip_model.ckpt` – region encoder
   * `cls_model.ckpt`  – risk-gated classifier

2. **Evaluate a single CWE**

   ```bash
   python main.py \
     --gpus 0 \
     --data_path "CWE/BigVul/CWE-119.pkl" \
     --clip_model_path checkpoints/clip_model.ckpt \
     --cls_model_path  checkpoints/cls_model.ckpt
   ```

---
