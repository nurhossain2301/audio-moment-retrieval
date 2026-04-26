---
license: cc-by-4.0
language:
- en
tags:
- multimodal
- audio-retrieval
- moment-retrieval
size_categories:
- 1K<n<10K
task_categories:
- audio-text-to-text
---

# CASTELLA CLAP features
This repository contains audio and text features of [CASTELLA dataset](https://arxiv.org/abs/2511.15131) extracted by CLAP.
- Using these features, we can reproduce the audio moments retrieval using CASTELLA, which is used in [lighthouse](https://github.com/line/lighthouse).
- Please also check [demo page](https://h-munakata.github.io/CASTELLA-demo/).

## How to Download?
Run the following script:
```python
from huggingface_hub import snapshot_download

repo_id = "lighthouse-emnlp2024/CASTELLA_CLAP_features"
local_dir = "./"

downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns="*.tar.gz",
)
```

## How to Use on Lighthouse
The `.tar.gz` files should be decompressed by following shell commands:
```bash
mkdir -p {LIGHTHOUSE_PATH}/features/castella/clap
mkdir -p {LIGHTHOUSE_PATH}/features/castella/clap_text
tar -zxvf clap.tar.gz -C {LIGHTHOUSE_PATH}/features/castella/clap
tar -zxvf clap_text.tar.gz -C {LIGHTHOUSE_PATH}/features/castella/clap_text
```



## Citation
```bibtex
@article{munakata2025castella,
  title={CASTELLA: Long Audio Dataset with Captions and Temporal Boundaries},
  author={Munakata, Hokuto and Takehiro, Imamura and Nishimura, Taichi and Komatsu, Tatsuya},
  journal={arXiv preprint arXiv:2511.15131},
  year={2025},
}
```# audio-moment-retrieval
