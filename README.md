<div align="center">
  <h3 align="center"><strong>Rethinking Event-Based Object Detection through Representation-Level Temporal Aggregation and Model-Level Hypergraph Reasoning </strong></h3>
    <p align="center">
    <a>Meisen Wang</a><sup>1</sup>&nbsp;&nbsp;
    <a>Hao Deng</a><sup>1</sup>&nbsp;&nbsp;
    <a>Wei Bao</a><sup>2</sup>&nbsp;&nbsp;
    <a>MaYuanxiao</a><sup>3</sup>&nbsp;&nbsp;
    <a>Chengjie Wang</a><sup>4</sup>&nbsp;&nbsp;
    <a>Zhiqiang Tian</a><sup>1</sup>&nbsp;&nbsp;
    <a>Shaoyi Du</a><sup>1</sup>&nbsp;&nbsp;
    <a>Siqi Li</a><sup>2</sup>&nbsp;&nbsp;
    <br>
    <sup>1</sup>Xi'an Jiaotong University&nbsp;&nbsp;&nbsp;
    <sup>2</sup>Tsinghua University&nbsp;&nbsp;&nbsp;
    <sup>3</sup>China Mobile System Integration&nbsp;&nbsp;&nbsp;
    <sup>4</sup>Inner Mongolia Agricultural University&nbsp;&nbsp;&nbsp;
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2605.08825">
    <img src="https://img.shields.io/badge/arXiv-2605.08825-b31b1b.svg" alt="arXiv:2605.08825">
  </a>

  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=zhiwen-xdu.ScaleEvent&left_color=gray&right_color=purple">
  </a>
</p>

### 📋 To-Do List
- [ ] [around 2026.6.8] Release the code, data and model

### ![image](https://github.com/user-attachments/assets/1ae19de2-b18b-4b0d-a206-19f0666757fb) About
Ev-DTAD is a novel event-based object detection framework that combines compact temporal event encoding with temporal-relational feature reasoning. Our approach introduces HTA-RGB, a hierarchical temporal aggregation representation, and FHTF, a frequency-aware hypergraph temporal fusion module, to jointly capture intra-/inter-window event dynamics and high-order feature dependencies.

## 🏆 Main Results


## 😀 Quick Start
### ⚙️ 1. Installation & Requirements

The package was tested only under Linux systems.

#### 1.1 Environment

The development environment is based on the
`pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` Docker container. You can set
up your environment using either Docker or Conda.

##### Option 1: Docker Setup

If using Docker (`pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime`), create a
Python virtual environment inside the container to avoid package conflicts:

```bash
python3 -m venv --system-site-packages ~/.venv/ev-dtad
source ~/.venv/ev-dtad/bin/activate
```

##### Option 2: Conda Setup

Alternatively, you can create a Conda environment using our provided
configuration:

```bash
conda env create -f contrib/conda_env.yml
conda activate ev-dtad
```

#### 1.2 Requirements

1. COCO evaluation metrics by Prophesee `psee_adt`.
Install the bundled package from `psee_adt-master`.

2. Python dependencies:
```bash
pip install -r requirements.txt
```

#### Installation

```bash
pip install -e .
```

For information about default data and output directory paths, refer to
[Default Paths and Environment Variables](#default-paths-and-environment-variables).

NOTE: It is recommended to increase the file descriptor limit before running
the training (see [File Descriptors Limit](#file-descriptors-limit)).
Otherwise, the training is likely to fail when using multiple data workers.

### 💾 2. Dataset Preparation

Due to license restrictions we are unable to distribute pre-processed datasets.
Therefore, the datasets need to be manually downloaded and pre-processed.


#### 2.1 Dataset Download

The Gen1 Prophesee dataset can be downloaded from
[this link](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/).

The Gen4/1MPX Prophesee dataset can be downloaded from
[this link](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).

The eTraM Prophesee dataset can be downloaded from
[this link](https://docs.google.com/forms/d/e/1FAIpQLSfH2LI5oqWWfose-pBC3dsbaAMvRQuv0BI93njV_5wQjYx83w/viewform?pli=1).

#### 2.2 Dataset Pre-processing

To preprocess the datasets into a HTA format, the `scripts/data/psee_to_htargb.py` script can be used.

To pre-process the Gen1 dataset, one can use the following command:
```bash
python3 scripts/data/psee_to_htargb.py \
      --dataset dataset_name \
      --src_root SRC \
      --out_root DST \
      --window_us 50000 \
      --clean_output
```

where
  - `dataset_name` specifies the target dataset type. It can be set to
     `gen1`, `gen4`, or `etram`
  - `SRC/` is a path to the original Prophesee dataset
  - `DST/` is a path where the pre-processed HTA-RGB dataset will be stored
  - `--window_us 50000` is a 50 ms time window used for event-to-frame conversion
  - `--clean_output` removes existing generated images and annotation files
     before running the pre-processing
  - when `dataset_name` is set to `etram`, the script uses the Gen4-style
     pre-processing entrance


### 🚀 3. Training Models From Scratch

The training of the Ev-DTAD models is staged:
1. At the first stage, simple RT-DETR models are trained on random EBC video
   frames.
2. At the second stage, the Ev-DTAD Memory modules are trained on EBC videos,
   using RT-DETR from stage 1 as an object detection backbone.


##### 3.1 Training RT-DETR models

`Ev-DTAD` provides several scripts to train the RT-DETR models:

```bash
# Gen1 Models
scripts/train/gen1/frame_detection_rtdetr/train_gen1_coco_rtdetr_presnet18.py
scripts/train/gen1/frame_detection_rtdetr/train_gen1_coco_rtdetr_presnet50.py

# Gen4/1MPX Models
scripts/train/gen4/frame_detection_rtdetr/train_gen4_coco_rtdetr_presnet18.py
scripts/train/gen4/frame_detection_rtdetr/train_gen4_coco_rtdetr_presnet50.py

# eTraM Models
scripts/train/etram/frame_detection_rtdetr/train_etram_coco_rtdetr_presnet18.py
scripts/train/etram/frame_detection_rtdetr/train_etram_coco_rtdetr_presnet50.py
```

Each of these scripts defines a training configuration and calls the training
routine. Feel free to examine and modify those scripts as needed.

To train an RT-DETR PResNet-18 model on the Gen1 dataset, simply run
```bash
python3 scripts/train/gen1/frame_detection_rtdetr/train_gen1_coco_rtdetr_presnet18.py
```

Once complete, the model will be saved under:
```
outdir/gen1/frame_rtdetr_presnet18/model_m(frame-detection-rtdetr)_default/
```

The output directory can be configured via environment variables (cf.
[Default Paths and Environment Variables](#default-paths-and-environment-variables)
).

Refer to [Model Directory Structure](#evlearn-model-structure) for details
on the directory contents.


##### 3.2 Training Ev-DTAD models

The Ev-DTAD models can be trained with the following scripts:

```bash
# Gen1 Models
scripts/train/gen1/video_detection_evdtad/train_gen1_coco_evdtad_presnet18.py
scripts/train/gen1/video_detection_evdtad/train_gen1_coco_evdtad_presnet50.py

# Gen4/1MPX Models
scripts/train/gen4/video_detection_evdtad/train_gen4_coco_evdtad_presnet18.py
scripts/train/gen4/video_detection_evdtad/train_gen4_coco_evdtad_presnet50.py

# eTraM Models
scripts/train/etram/video_detection_evdtad/train_etram_coco_evdtad_presnet18.py
scripts/train/etram/video_detection_evdtad/train_etram_coco_evdtad_presnet50.py
```

These scripts expect to find the pre-trained RT-DETR models from the previous
step under `outdir/models`. Please place the pre-trained models there (move
copy entire model directory), or modify the script's `TRANSFER_PATH`
variable to choose another location.

For example, the Gen1 Ev-DTAD PResNet-18 script, expects to find a
pre-trained RT-DETR PResNet-18 model under
```
outdir/models/gen1/frame_rtdetr_presnet18
```

Once the pre-trained RT-DETR model is placed in that location, the Ev-DTAD
training can be started:
```bash
python3 scripts/train/gen1/video_detection_evdtad/train_gen1_coco_evdtad_presnet18.py

```

After the training is complete, the trained model will be in:
```
outdir/gen1/video_evdtad_presnet18/model_m(vcf-detection-evdtad)_default/
```

Refer to [Model Directory Structure](#evlearn-model-structure) for details
on the directory contents.


#### ⭐️ 4. Evaluation

To evaluate the COCO mAP metrics `Ev-DTAD` provides script:
```
scripts/eval_model_video.py
```

To evaluate the performance of the Ev-DTAD model, one can run
```bash
python3 scripts/eval_model_video.py PATH_TO_MODEL_DIRECTORY --data-name video
```
where `PATH_TO_MODEL_DIRECTORY` is a path where the trained Ev-DTAD model is
saved. When the evaluation is complete, the COCO scores will be printed to the
terminal and saved in the model's `evals/` subdirectory
(cf. [Model Directory Structure](#evlearn-model-structure)).

### 📚 Citation
If you use ScaleEvent in your research, please use the following BibTeX entry.

```
@article{wang2026rethinking,
  title={Rethinking Event-Based Object Dtection through Representation-Level Temporal Aggregation and Model-Level Hypergraph Reasoning},
  author={Wang, Meisen and Deng, Hao and Bao, Wei and Yuanxiao, Ma and Wang, Chengjie and Tian, Zhiqiang and Du, Shaoyi and Li, Siqi},
  journal={arXiv preprint arXiv:2605.08825},
  year={2026}
}
```
