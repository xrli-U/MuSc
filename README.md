# ✨MuSc (ICLR 2024)✨

**This is an official PyTorch implementation for "MuSc : Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images" (MuSc)**

Authors:  [Xurui Li](https://github.com/xrli-U)<sup>1*</sup> | [Ziming Huang](https://github.com/ZimingHuang1)<sup>1*</sup> | [Feng Xue](https://xuefeng-cvr.github.io/)<sup>3</sup> | [Yu Zhou](https://github.com/zhouyu-hust)<sup>1,2</sup>

Institutions: <sup>1</sup>Huazhong University of Science and Technology | <sup>2</sup>Wuhan JingCe Electronic Group Co.,LTD | <sup>3</sup>University of Trento

### 🧐  [Arxiv](https://arxiv.org/pdf/2401.16753.pdf) | [OpenReview](https://openreview.net/forum?id=AHgc5SMdtd)

### 📖 Chinese [README](./README_cn.md)

## <a href='#all_catelogue'>**Go to Catalogue**</a>

## 🙈TODO list:
- ⬜️ Using some strategies to reduce the inference time per image from 955.3ms to **249.8ms**.
- ⬜️ Compatibility with more industrial datasets.
- ⬜️ Compatibility with more visual backbones, e.g. [Vision Mamba](https://github.com/hustvl/Vim).


## 📣Updates:
***04/11/2024***
1. The comparisons with the zero/few-shot methods in CVPR 2024 have been added to <a href='#compare_sota'>Compare with SOTA k-shot Methods.</a>
2. Fixed some bugs in `models/backbone/_backbones.py`.

***03/22/2024***
1. The supported codes for [BTAD](https://ieeexplore.ieee.org/abstract/document/9576231) dataset are provided.
2. Some codes are modified to support larger *batch_size*.
3. Some codes are optimized to obtain faster speeds.
4. <a href='#results_backbones'>Results of different backbones</a> in MVTec AD, VisA and BTAD datasets are provided.
5. <a href='#results_datasets'>The detailed results of different datasets</a> are provided.
6. <a href='#inference_time'>The inference time of different backbones</a> is provided.
7. <a href='#compare_sota'>The comparisons with SOTA zero/few-shot methods</a> are provided. This table will be updated continuously.
8. We summarize the <a href='#FAQ'> frequently asked questions </a> from users when using MuSc, and give the answers.
9. We add [README](./README_cn.md) in Chinese.

***02/01/2024***

Initial commits:

1. The complete code of our method **MuSc** in [paper](https://arxiv.org/pdf/2401.16753.pdf) is released.
2. This code is compatible with image encoder (ViT) of [CLIP](https://github.com/mlfoundations/open_clip) and ViT pre-trained with [DINO](https://github.com/facebookresearch/dino)/[DINO_v2](https://github.com/facebookresearch/dinov2).

<span id='compare_sota'/>

## 🎖️Compare with SOTA *k*-shot methods <a href='#all_catelogue'>[Go to Catalogue]</a>
We will **continuously update** the following table to compare our MuSc with the newest zero-shot and few-shot methods.
"-" indicates that the authors did not measure this metric in their paper.

### MVTec AD

|                                          |                    |         | Classification |            |        | Segmentation |             |         |          |
| :--------------------------------------: | :----------------: | :-----: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|                 Methods                  |       Venue        | Setting |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|                MuSc(ours)                |     ICLR 2024      | 0-shot  |      97.8      |    97.5    |  99.1  |     97.3     |    62.6     |  62.7   |   93.8   |
| [RegAD](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_18) |     ECCV 2022      | 4-shot  |      89.1      |    92.4    |  94.9  |     96.2     |    51.7     |  48.3   |   88.0   |
| [GraphCore](https://openreview.net/forum?id=xzmqxHdZAwO) |     ICLR 2023      | 4-shot  |      92.9      |     -      |   -    |     97.4     |      -      |    -    |    -     |
| [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) |     CVPR 2023      | 0-shot  |      91.8      |    92.9    |  96.5  |     85.1     |    31.7     |    -    |   64.6   |
| [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) |     CVPR 2023      | 4-shot  |      95.2      |    94.7    |  97.3  |     96.2     |    51.7     |    -    |   88.0   |
| [APRIL-GAN](https://arxiv.org/pdf/2305.17382.pdf) | CVPR Workshop 2023 | 0-shot  |      86.1      |    90.4    |  93.5  |     87.6     |    43.3     |  40.8   |   44.0   |
| [APRIL-GAN](https://arxiv.org/pdf/2305.17382.pdf) | CVPR Workshop 2023 | 4-shot  |      92.8      |    92.8    |  96.3  |     95.9     |    56.9     |  54.5   |   91.8   |
| [FastRecon](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf) |     ICCV 2023      | 4-shot  |      94.2      |     -      |   -    |     97.0     |      -      |    -    |    -     |
| [ACR](https://proceedings.neurips.cc/paper_files/paper/2023/file/8078e8c3055303a884ffae2d3ea00338-Paper-Conference.pdf) |    NeurIPS 2023    | 0-shot  |      85.8      |    91.3    |  92.9  |     92.5     |    44.2     |  38.9   |   72.7   |
| [RegAD+Adversarial Loss](https://papers.bmvc2023.org/0202.pdf) |     BMVC 2023      | 8-shot  |      91.9      |     -      |   -    |     96.9     |      -      |    -    |    -     |
| [PACKD](https://papers.bmvc2023.org/0259.pdf) |     BMVC 2023      | 8-shot  |      95.3      |     -      |   -    |     97.3     |      -      |    -    |    -     |
| [PromptAD](https://openaccess.thecvf.com/content/WACV2024/papers/Li_PromptAD_Zero-Shot_Anomaly_Detection_Using_Text_Prompts_WACV_2024_paper.pdf) |     WACV 2024      | 0-shot  |      90.8      |     -      |   -    |     92.1     |    36.2     |    -    |   72.8   |
| [AnomalyCLIP](https://openreview.net/forum?id=buC4E91xZE) |     ICLR 2024      | 0-shot  |      91.5      |     -      |  96.2  |     91.1     |      -      |    -    |   81.4   |
| [InCTRL](https://arxiv.org/pdf/2403.06495.pdf) |     CVPR 2024      | 8-shot  |      95.3      |     -      |   -    |      -       |      -      |    -    |    -     |
| [MVFA-AD](https://arxiv.org/pdf/2403.12570.pdf) |     CVPR 2024      | 4-shot  |      96.2      |     -      |   -    |     96.3     |      -      |    -    |    -     |
| [PromptAD](https://arxiv.org/pdf/2404.05231.pdf) |     CVPR 2024      | 4-shot  |      96.6      |     -      |   -    |     96.5     |      -      |    -    |    -     |

### VisA

|                                          |                    |         | Classification |            |        | Segmentation |             |         |          |
| :--------------------------------------: | :----------------: | :-----: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|                 Methods                  |       Venue        | Setting |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|                MuSc(ours)                |     ICLR 2024      | 0-shot  |      92.8      |    89.5    |  93.5  |     98.8     |    48.8     |  45.1   |   92.7   |
| [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) |     CVPR 2023      | 0-shot  |      78.1      |    79.0    |  81.2  |     79.6     |    14.8     |    -    |   56.8   |
| [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) |     CVPR 2023      | 4-shot  |      87.3      |    84.2    |  88.8  |     97.2     |    47.0     |    -    |   87.6   |
| [APRIL-GAN](https://arxiv.org/pdf/2305.17382.pdf) | CVPR Workshop 2023 | 0-shot  |      78.0      |    78.7    |  81.4  |     94.2     |    32.3     |  25.7   |   86.8   |
| [APRIL-GAN](https://arxiv.org/pdf/2305.17382.pdf) | CVPR Workshop 2023 | 4-shot  |      92.6      |    88.4    |  94.5  |     96.2     |    40.0     |  32.2   |   90.2   |
| [PACKD](https://papers.bmvc2023.org/0259.pdf) |     BMVC 2023      | 8-shot  |      87.5      |     -      |   -    |     97.9     |      -      |    -    |    -     |
| [AnomalyCLIP](https://openreview.net/forum?id=buC4E91xZE) |     ICLR 2024      | 0-shot  |      82.1      |     -      |  85.4  |     95.5     |      -      |    -    |   87.0   |
| [InCTRL](https://arxiv.org/pdf/2403.06495.pdf) |     CVPR 2024      | 8-shot  |      88.7      |     -      |   -    |      -       |      -      |    -    |    -     |
| [PromptAD](https://arxiv.org/pdf/2404.05231.pdf) |     CVPR 2024      | 4-shot  |      89.1      |     -      |   -    |     97.4     |      -      |    -    |    -     |

<span id='all_catelogue'/>

## 📖Catalogue

* <a href='#abstract'>1. Abstract</a>
* <a href='#setup'>2. Environment setup</a>
* <a href='#datasets'>3. Datasets download</a>
  * <a href='#datatets_mvtec_ad'>MVTec AD</a>
  * <a href='#datatets_visa'>VisA</a>
  * <a href='#datatets_btad'>BTAD</a>
* <a href='#run_musc'>4. Run MuSc</a>
* <a href='#rscin'>5. Run RsCIN</a>
* <a href='#results_datasets'>6. Results of different datasets</a>
* <a href='#results_backbones'>7. Results of different backbones</a>
* <a href='#inference_time'>8. Inference time</a>
* <a href='#FAQ'>9. Frequently Asked Questions</a>
* <a href='#citation'>10. Citation</a>
* <a href='#thanks'>11. Thanks</a>
* <a href='#license'>12. License</a>

<span id='abstract'/>

## 👇Abstract: <a href='#all_catelogue'>[Back to Catalogue]</a>

This paper studies zero-shot anomaly classification (AC) and segmentation (AS) in industrial vision. We reveal that the abundant normal and abnormal cues implicit in unlabeled test images can be exploited for anomaly determination, which is ignored by prior methods. Our key observation is that for the industrial product images, the normal image patches could find a relatively large number of similar patches in other unlabeled images, while the abnormal ones only have a few similar patches. 

We leverage such a discriminative characteristic to design a novel zero-shot AC/AS method by Mutual Scoring (MuSc) of the unlabeled images, which does not need any training or prompts. Specifically, we perform Local Neighborhood Aggregation with Multiple Degrees (**LNAMD**) to obtain the patch features that are capable of representing anomalies in varying sizes. Then we propose the Mutual Scoring Mechanism (**MSM**) to leverage the unlabeled test images to assign the anomaly score to each other. Furthermore, we present an optimization approach named Re-scoring with Constrained Image-level Neighborhood (**RsCIN**) for image-level anomaly classification to suppress the false positives caused by noises in normal images.

The superior performance on the challenging MVTec AD and VisA datasets demonstrates the effectiveness of our approach. Compared with the state-of-the-art zero-shot approaches, MuSc achieves a $\textbf{21.1}$\% PRO absolute gain (from 72.7\% to 93.8\%) on MVTec AD, a $\textbf{19.4}$\% pixel-AP gain and a $\textbf{14.7}$\% pixel-AUROC gain on VisA. In addition, our zero-shot approach outperforms most of the few-shot approaches and is comparable to some one-class methods.

![pipline](./assets/pipeline.png) 

## 😊Compare with other 0-shot methods

![Compare_0](./assets/compare_zero_shot.png) 

## 😊Compare with other 4-shot methods

![Compare_4](./assets/compare_few_shot.png) 

<span id='setup'/>

## 🎯Setup: <a href='#all_catelogue'>[Back to Catalogue]</a>

### Environment:

- Python 3.8
- CUDA 11.7
- PyTorch 2.0.1

Clone the repository locally:

```
git clone https://github.com/xrli-U/MuSc.git
```

Create virtual environment:

```
conda create --name musc python=3.8
conda activate musc
```

Install the required packages:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

<span id='datasets'/>

## 👇Datasets Download: <a href='#all_catelogue'>[Back to Catalogue]</a>

Put all the datasets in `./data` folder.

<span id='datatets_mvtec_ad'/>

### [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

```
data
|---mvtec_anomaly_detection
|-----|-- bottle
|-----|-----|----- ground_truth
|-----|-----|----- test
|-----|-----|----- train
|-----|-- cable
|-----|--- ...
```

<span id='datatets_visa'/>

### [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

```
data
|----visa
|-----|-- split_csv
|-----|-----|--- 1cls.csv
|-----|-----|--- ...
|-----|-- candle
|-----|-----|--- Data
|-----|-----|-----|----- Images
|-----|-----|-----|--------|------ Anomaly 
|-----|-----|-----|--------|------ Normal 
|-----|-----|-----|----- Masks
|-----|-----|-----|--------|------ Anomaly 
|-----|-----|--- image_anno.csv
|-----|-- capsules
|-----|--- ...
```

VisA dataset need to be preprocessed to separate the train set from the test set.

```
python ./datasets/visa_preprocess.py
```

<span id='datatets_btad'/>

### [BTAD](https://github.com/pankajmishra000/VT-ADL)

```
data
|---btad
|-----|--- 01
|-----|-----|----- ground_truth
|-----|-----|----- test
|-----|-----|----- train
|-----|--- 02
|-----|--- ...
```

<span id='run_musc'/>

## 💎Run MuSc: <a href='#all_catelogue'>[Back to Catalogue]</a>

We provide two ways to run our code.

### python

```
python examples/musc_main.py
```
Follow the configuration in `./configs/musc.yaml`.

### script

```
sh scripts/musc.sh
```
The configuration in the script `musc.sh` takes precedence.

The key arguments of the script are as follows:

- `--device`: GPU_id.
- `--data_path`: The directory of datasets.
- `--dataset_name`: Dataset name.
- `--class_name`: Category to be tested. If the parameter is set to `ALL`, all the categories are tested.
- `--backbone_name`: Feature exractor name. Our code is compatible with CLIP, DINO and DINO_v2. For more details, see `configs/musc.yaml`.
- `--pretrained`: Pretrained CLIP model. `openai`, `laion400m_e31`, and `laion400m_e32` are optional.
- `--feature_layers`: The layers for extracting features in backbone(ViT).
- `--img_resize`: The size of the image inputted into the model.
- `--divide_num`: The number of subsets the whole test set is divided into.
- `--r_list`: The aggregation degrees of our LNAMD module.
- `--output_dir`: The directory that saves the anomaly prediction maps and metrics. This directory will be automatically created.
- `--vis`: Whether to save the anomaly prediction maps.
- `--vis_type`: Choose between `single_norm` and `whole_norm`. This means whether to normalize a single anomaly map or all of them together when visualizing.
- `--save_excel`: Whether to save anomaly classification and segmentation results (metrics).

<span id='rscin'/>

## 💎Classification optimization (RsCIN): <a href='#all_catelogue'>[Back to Catalogue]</a>

We provide additional code in `./models/RsCIN_features` folder to optimize the classification results of other methods using our RsCIN module. We use **ViT-large-14-336 of CLIP** to extract the image features of the MVTec AD and VisA datasets and store them in `mvtec_ad_cls.dat` and `visa_cls.dat` respectively. We show how to use them in `./models/RsCIN_features/RsCIN.py`.

### Example

Before using our RsCIN module, move `RsCIN.py`, `mvtec_ad_cls.dat` and `visa_cls.dat` to your project directory.

```
import numpy as np
from RsCIN import Mobile_RsCIN

classification_results = np.random.rand(83) # the classification results of your method.
dataset_name = 'mvtec_ad' # dataset name
class_name = 'bottle' # category name in the above dataset
optimized_classification_results = Mobile_RsCIN(classification_results, dataset_name=dataset_name, class_name=class_name)
```

The `optimized_classification_results` are the anomaly classification scores optimized by our RsCIN module.

### Apply to the custom dataset

You can extract the image features of each image in the custom dataset, and store them in the variable `cls_tokens`.
The multiple window sizes in the Multi-window Mask Operation can be adjusted by the value of `k_list`.

```
import numpy as np
from RsCIN import Mobile_RsCIN

classification_results = np.random.rand(83) # the classification results of your method.
cls_tokens = np.random.rand(83, 768)  # shape[N, C] the image features, N is the number of images
k_list = [2, 3] # the multiple window sizes in the Multi-window Mask Operation
optimized_classification_results = Mobile_RsCIN(classification_results, k_list=k_list, cls_tokens=cls_tokens)
```

<span id='results_datasets'/>

## 🎖️Results of different datasets: <a href='#all_catelogue'>[Back to Catalogue]</a>

All the results are implemented by the default settings in our paper.

### MVTec AD

|            | Classification |            |        | Segmentation |             |         |          |
| :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|  Category  |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|   bottle   |     99.92      |   99.21    | 99.98  |    98.48     |    79.17    |  83.04  |  96.10   |
|   cable    |     98.99      |   97.30    | 99.42  |    95.76     |    60.97    |  57.70  |  89.62   |
|  capsule   |     96.45      |   94.88    | 99.30  |    98.96     |    49.80    |  48.45  |  95.49   |
|   carpet   |     99.88      |   99.44    | 99.96  |    99.45     |    73.33    |  76.05  |  97.58   |
|    grid    |     98.66      |   96.49    | 99.54  |    98.16     |    43.94    |  38.24  |  93.92   |
|  hazelnut  |     99.61      |   98.55    | 99.79  |    99.38     |    73.41    |  73.28  |  92.24   |
|  leather   |     100.0      |   100.0    | 100.0  |    99.72     |    62.84    |  64.47  |  98.74   |
| metal_nut  |     96.92      |   97.38    | 99.25  |    86.12     |    46.22    |  47.54  |  89.34   |
|    pill    |     96.24      |   95.89    | 99.31  |    97.47     |    65.54    |  67.25  |  98.01   |
|   screw    |     82.17      |   88.89    | 90.88  |    98.77     |    41.87    |  36.12  |  94.40   |
|    tile    |     100.0      |   100.0    | 100.0  |    97.90     |    74.71    |  78.90  |  94.64   |
| toothbrush |     100.0      |   100.0    | 100.0  |    99.53     |    70.19    |  67.79  |  95.48   |
| transistor |     99.42      |   95.00    | 99.19  |    91.38     |    59.24    |  58.40  |  77.21   |
|    wood    |     98.51      |   98.33    | 99.52  |    97.24     |    68.64    |  74.75  |  94.50   |
|   zipper   |     99.84      |   99.17    | 99.96  |    98.40     |    62.48    |  61.89  |  94.46   |
|    mean    |     97.77      |   97.37    | 99.07  |    97.11     |    62.16    |  62.26  |  93.45   |

### VisA

|            | Classification |            |        | Segmentation |             |         |          |
| :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|  Category  |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|   candle   |     96.55      |   91.26    | 96.45  |    99.36     |    39.56    |  28.36  |  97.62   |
|  capsules  |     88.62      |   86.43    | 93.77  |    98.71     |    50.85    |  43.90  |  88.20   |
|   cashew   |     98.54      |   95.57    | 99.30  |    99.33     |    74.88    |  77.63  |  94.30   |
| chewinggum |     98.42      |   96.45    | 99.30  |    99.54     |    61.33    |  61.21  |  88.39   |
|   fryum    |     98.64      |   97.44    | 99.43  |    99.43     |    58.13    |  50.43  |  94.38   |
| macaroni1  |     89.33      |   82.76    | 88.64  |    99.51     |    21.90    |  15.25  |  96.37   |
| macaroni2  |     68.03      |   69.96    | 67.37  |    97.14     |    11.06    |  3.91   |  88.84   |
|    pcb1    |     89.28      |   84.36    | 89.89  |    99.50     |    80.49    |  88.36  |  92.76   |
|    pcb2    |     93.20      |   88.66    | 94.46  |    97.39     |    34.38    |  21.86  |  86.06   |
|    pcb3    |     93.52      |   86.92    | 93.48  |    98.05     |    40.23    |  41.03  |  92.32   |
|    pcb4    |     98.43      |   92.89    | 98.47  |    98.70     |    46.38    |  44.72  |  92.66   |
| pipe_fryum |     98.34      |   96.04    | 99.16  |    99.40     |    67.56    |  67.90  |  97.32   |
|    mean    |     92.57      |   89.06    | 93.31  |    98.71     |    48.90    |  45.38  |  92.43   |

### BTAD

|          | Classification |            |        | Segmentation |             |         |          |
| :------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
| Category |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|    01    |     98.74      |   97.96    | 99.53  |    97.49     |    59.73    |  58.76  |  85.05   |
|    02    |     90.23      |   95.38    | 98.41  |    95.36     |    58.20    |  55.16  |  68.64   |
|    03    |     99.52      |   88.37    | 95.62  |    99.20     |    55.64    |  57.53  |  96.62   |
|   mean   |     96.16      |   93.90    | 97.85  |    97.35     |    57.86    |  57.15  |  83.43   |

<span id='results_backbones'/>

## 🎖️Results of different backbones: <a href='#all_catelogue'>[Back to Catalogue]</a>

The default backbone (feature extractor) in our paper is ViT-large-14-336 of CLIP.
We also provide the supported codes for other image encoder of CLIP, DINO and DINO_v2.
For more details, see `configs/musc.yaml`.

### MVTec AD

|                   |              |            | Classification |            |        | Segmentation |             |         |          |
| :---------------: | :----------: | :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|     Backbones     | Pre-training | image size |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|     ViT-B-32      |     CLIP     |    256     |     87.99      |   92.31    | 94.38  |    93.08     |    42.06    |  37.21  |  72.62   |
|     ViT-B-32      |     CLIP     |    512     |     89.91      |   92.72    | 95.12  |    95.73     |    53.32    |  52.33  |  83.72   |
|     ViT-B-16      |     CLIP     |    256     |     92.78      |   93.98    | 96.59  |    96.21     |    52.48    |  50.23  |  87.00   |
|     ViT-B-16      |     CLIP     |    512     |     94.20      |   95.20    | 97.34  |    97.09     |    61.24    |  61.45  |  91.67   |
| ViT-B-16-plus-240 |     CLIP     |    240     |     94.77      |   95.43    | 97.60  |    96.26     |    52.23    |  50.27  |  87.70   |
| ViT-B-16-plus-240 |     CLIP     |    512     |     95.69      |   96.50    | 98.11  |    97.28     |    60.71    |  61.29  |  92.14   |
|     ViT-L-14      |     CLIP     |    336     |     96.06      |   96.65    | 98.25  |    97.24     |    59.41    |  58.10  |  91.69   |
|     ViT-L-14      |     CLIP     |    518     |     95.94      |   96.32    | 98.30  |    97.42     |    63.06    |  63.67  |  92.92   |
|   ViT-L-14-336    |     CLIP     |    336     |     96.40      |   96.44    | 98.30  |    97.03     |    57.51    |  55.44  |  92.18   |
|   ViT-L-14-336    |     CLIP     |    518     |     97.77      |   97.37    | 99.07  |    97.11     |    62.16    |  62.26  |  93.45   |
|  dino_vitbase16   |     DINO     |    256     |     89.39      |   93.77    | 95.37  |    95.83     |    54.02    |  52.84  |  84.24   |
|  dino_vitbase16   |     DINO     |    512     |     94.11      |   96.13    | 97.26  |    97.78     |    62.07    |  63.20  |  92.49   |
|   dinov2_vitb14   |   DINO_v2    |    336     |     95.67      |   96.80    | 97.95  |    97.74     |    60.23    |  59.45  |  93.84   |
|   dinov2_vitb14   |   DINO_v2    |    518     |     96.31      |   96.87    | 98.32  |    98.07     |    64.65    |  65.31  |  95.59   |
|   dinov2_vitl14   |   DINO_v2    |    336     |     96.84      |   97.45    | 98.68  |    98.17     |    61.77    |  61.21  |  94.62   |
|   dinov2_vitl14   |   DINO_v2    |    518     |     97.08      |   97.13    | 98.82  |    98.34     |    66.15    |  67.39  |  96.16   |


### VisA

|                   |              |            | Classification |            |        | Segmentation |             |         |          |
| :---------------: | :----------: | :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|     Backbones     | Pre-training | image size |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|     ViT-B-32      |     CLIP     |    256     |     72.95      |   76.90    | 77.68  |    89.30     |    25.93    |  20.68  |  50.95   |
|     ViT-B-32      |     CLIP     |    512     |     77.82      |   80.20    | 81.01  |    96.06     |    34.72    |  30.20  |  73.08   |
|     ViT-B-16      |     CLIP     |    256     |     81.44      |   80.86    | 83.84  |    95.97     |    36.72    |  31.81  |  73.48   |
|     ViT-B-16      |     CLIP     |    512     |     86.48      |   84.12    | 88.05  |    97.98     |    42.21    |  37.29  |  85.10   |
| ViT-B-16-plus-240 |     CLIP     |    240     |     82.62      |   81.61    | 85.05  |    96.11     |    37.84    |  33.43  |  72.37   |
| ViT-B-16-plus-240 |     CLIP     |    512     |     86.72      |   84.22    | 89.41  |    97.95     |    43.27    |  37.68  |  83.52   |
|     ViT-L-14      |     CLIP     |    336     |     88.38      |   85.23    | 89.77  |    98.32     |    44.67    |  40.42  |  87.80   |
|     ViT-L-14      |     CLIP     |    518     |     90.86      |   87.75    | 91.66  |    98.45     |    45.74    |  42.09  |  89.93   |
|   ViT-L-14-336    |     CLIP     |    336     |     88.61      |   85.31    | 90.00  |    98.53     |    45.10    |  40.92  |  89.35   |
|   ViT-L-14-336    |     CLIP     |    518     |     92.57      |   89.06    | 93.31  |    98.71     |    48.90    |  45.38  |  92.43   |
|  dino_vitbase16   |     DINO     |    256     |     78.21      |   80.12    | 81.11  |    95.74     |    36.81    |  32.84  |  70.21   |
|  dino_vitbase16   |     DINO     |    512     |     84.11      |   83.52    | 85.91  |    97.74     |    42.86    |  38.27  |  83.00   |
|   dinov2_vitb14   |   DINO_v2    |    336     |     87.65      |   86.24    | 88.51  |    97.80     |    41.68    |  37.06  |  85.01   |
|   dinov2_vitb14   |   DINO_v2    |    518     |     90.25      |   87.48    | 90.86  |    98.66     |    45.56    |  41.23  |  91.80   |
|   dinov2_vitl14   |   DINO_v2    |    336     |     90.18      |   88.47    | 90.56  |    98.38     |    43.84    |  38.74  |  88.38   |
|   dinov2_vitl14   |   DINO_v2    |    518     |     91.73      |   89.20    | 92.27  |    98.78     |    47.12    |  42.79  |  92.40   |


### BTAD

|                   |              |            | Classification |            |        | Segmentation |             |         |          |
| :---------------: | :----------: | :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|     Backbones     | Pre-training | image size |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|     ViT-B-32      |     CLIP     |    256     |     92.19      |   95.55    | 98.47  |    96.74     |    43.98    |  35.70  |  68.56   |
|     ViT-B-32      |     CLIP     |    512     |     93.31      |   94.61    | 98.40  |    97.41     |    52.94    |  48.80  |  69.59   |
|     ViT-B-16      |     CLIP     |    256     |     92.44      |   91.00    | 97.31  |    97.45     |    55.27    |  52.19  |  72.68   |
|     ViT-B-16      |     CLIP     |    512     |     94.11      |   92.99    | 97.98  |    97.91     |    59.18    |  59.05  |  77.86   |
| ViT-B-16-plus-240 |     CLIP     |    240     |     92.86      |   93.99    | 97.96  |    97.68     |    54.81    |  51.33  |  73.47   |
| ViT-B-16-plus-240 |     CLIP     |    512     |     94.13      |   93.84    | 98.34  |    98.14     |    58.66    |  57.53  |  77.23   |
|     ViT-L-14      |     CLIP     |    336     |     92.74      |   93.21    | 97.71  |    97.84     |    56.60    |  55.94  |  77.01   |
|     ViT-L-14      |     CLIP     |    518     |     94.82      |   95.29    | 98.58  |    97.77     |    55.55    |  55.46  |  80.62   |
|   ViT-L-14-336    |     CLIP     |    336     |     95.11      |   94.48    | 98.53  |    97.42     |    56.75    |  55.23  |  79.63   |
|   ViT-L-14-336    |     CLIP     |    518     |     96.16      |   93.90    | 97.85  |    97.35     |    57.86    |  57.15  |  83.43   |
|  dino_vitbase16   |     DINO     |    256     |     93.63      |   95.66    | 98.66  |    97.55     |    52.16    |  49.25  |  72.86   |
|  dino_vitbase16   |     DINO     |    512     |     92.38      |   92.66    | 97.81  |    97.44     |    53.32    |  53.02  |  74.91   |
|   dinov2_vitb14   |   DINO_v2    |    336     |     93.60      |   91.65    | 97.19  |    98.08     |    63.28    |  65.32  |  74.35   |
|   dinov2_vitb14   |   DINO_v2    |    518     |     94.99      |   95.11    | 98.55  |    98.30     |    65.75    |  68.89  |  80.41   |
|   dinov2_vitl14   |   DINO_v2    |    336     |     94.15      |   92.64    | 97.61  |    98.19     |    63.86    |  66.03  |  76.33   |
|   dinov2_vitl14   |   DINO_v2    |    518     |     95.62      |   95.40    | 98.76  |    98.40     |    65.88    |  69.90  |  82.47   |

<span id='inference_time'/>

## ⌛Inference Time: <a href='#all_catelogue'>[Back to Catalogue]</a>

We show the inference time per image in the table below when using different backbones and image sizes.
The default setting for number of images in mutual scoring module is **200**, and GPU is NVIDIA RTX 3090.

|                   |              |            |                 |
| :---------------: | :----------: | :--------: | :-------------: |
|     Backbones     | Pre-training | image size | times(ms/image) |
|     ViT-B-32      |     CLIP     |    256     |      48.33      |
|     ViT-B-32      |     CLIP     |    512     |      95.74      |
|     ViT-B-16      |     CLIP     |    256     |      86.68      |
|     ViT-B-16      |     CLIP     |    512     |      450.5      |
| ViT-B-16-plus-240 |     CLIP     |    240     |      85.25      |
| ViT-B-16-plus-240 |     CLIP     |    512     |      506.4      |
|     ViT-L-14      |     CLIP     |    336     |      266.0      |
|     ViT-L-14      |     CLIP     |    518     |      933.3      |
|   ViT-L-14-336    |     CLIP     |    336     |      270.2      |
|   ViT-L-14-336    |     CLIP     |    518     |      955.3      |
|  dino_vitbase16   |     DINO     |    256     |      85.97      |
|  dino_vitbase16   |     DINO     |    512     |      458.5      |
|   dinov2_vitb14   |   DINO_v2    |    336     |      209.1      |
|   dinov2_vitb14   |   DINO_v2    |    518     |      755.0      |
|   dinov2_vitl14   |   DINO_v2    |    336     |      281.4      |
|   dinov2_vitl14   |   DINO_v2    |    518     |     1015.1      |

<span id='FAQ'/>

## 🙋🙋‍♂️Frequently Asked Questions: <a href='#all_catelogue'>[Back to Catalogue]</a>

Q: Why do large areas of high anomaly scores appear on normal images in the visualization?

A: In the visualization, in order to highlight abnormal areas, we adopt a single anomaly map normalization by default. Even if the overall response of the single map is low, a large number of highlighted areas will appear after normalization. Normalization of all the anomaly maps together can be achieved by adding the `vis_type` parameter to the shell script and setting it as `whole_norm`, or by modifying the `testing->vis_type` parameter in the `./configs/musc.yaml`.

Q: How to set the appropriate input image resolution ?

A: The image resolution `img_resize` input into the backbone is generally set to a multiple of the patch size of ViT.
The commonly used values are 224, 240, 256, 336, 512 and 518.
In the previous section <a href='#results_backbones'>*(jump)*</a>, we show the two input image resolutions commonly used by different feature extractors for reference.
The image resolution can be changed by modifying the 'img_resize' parameter in the shell script, or by modifying the `datasets->img_resize` parameter in the `./configs/musc.yaml` configuration file.

<span id='citation'/>

## Citation: <a href='#all_catelogue'>[Back to Catalogue]</a>
```
@inproceedings{Li2024MuSc,
  title={MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images},
  author={Li, Xurui and Huang, Ziming and Xue, Feng and Zhou, Yu},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

<span id='thanks'/>

## Thanks: <a href='#all_catelogue'>[Back to Catalogue]</a>

Our repo is built on [PatchCore](https://github.com/amazon-science/patchcore-inspection) and [APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN), thanks their clear and elegant code !

<span id='license'/>

## License: <a href='#all_catelogue'>[Back to Catalogue]</a>
MuSc is released under the **MIT Licence**, and is fully open for academic research and also allow free commercial usage. To apply for a commercial license, please contact yuzhou@hust.edu.cn.
