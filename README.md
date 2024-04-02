## <div align="center"> <i>CAMixerSR</i>: Only Details Need More “Attention” </div>

<p align="center">
<a href="https://arxiv.org/abs/2402.19289" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2402.19289-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/icandle/CAMixerSR/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<a href="https://colab.research.google.com/gist/icandle/404a89adbc264294dd77edacfd80f3b2/camixersr.ipynb" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

**Overview:** We propose ***CAMixerSR***, a new approach integrating *content-aware accelerating framework* and *token mixer design*, to pursue more efficient SR inference via assigning convolution for simple regions but window-attention for complex textures. It exhibits excellent generality and attains competitive results among state-of-the-art models with better complexity-performance trade-offs on large-image SR, lightweight SR, and omnidirectional-image SR.

<p align="center">
<img src="./figures/CAMixer.png" width=100% height=100% 
class="center">
</p>

This repository contains [PyTorch](https://pytorch.org/) implementation for ***CAMixerSR*** (CVPR 2024):

1. [Requirements](#%EF%B8%8F-requirements)
2. [Datasets](#-datasets)
3. [Test](#%EF%B8%8F-how-to-test)
4. [Results](#-results)
5. [Acknowledgments](#-acknowledgments)
6. [Citation](#-citation)
---

The main codes and pre-trained models have been uploaded. We are planning to replenish the README of CAMixerSR in a few weeks.
 
⚙️ Requirements
---
  
#### Dependencies
- [PyTorch >= 1.7](https://pytorch.org/) 
- [BasicSR == 1.4.2](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md)
- [einops](https://github.com/arogozhnikov/einops)
#### Installation
```
pip install -r requirements.txt
```


🎈 Datasets
---
#### Large-Image SR

*Training*: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

*Testing*: [F2K](https://drive.google.com/file/d/1jubeKExUrUE-VKpiTx9AAUVodJZvdkBC/view?usp=drive_link), [Test2K](https://drive.google.com/file/d/1HQtKhWrVSrUNh0bcka57BlhbZWCCB649/view?usp=drive_link), [Test4K](https://drive.google.com/file/d/1yBCRPHzcNzSX6xLsgFjgJuRlcVJWA4ZD/view?usp=sharing), [Test8K](https://drive.google.com/file/d/1CXVF61888zQRP8gkPGoX5LFD9IBcV-Sl/view?usp=drive_link) ([Google Drive](https://drive.google.com/drive/folders/1wSdB9GUa2IsYe5S8pHV-7d8dYdZA2wDp?usp=drive_link)/[Baidu Netdisk](https://pan.baidu.com/s/1IR90NxGRPajQLw9nFDflyA?pwd=nbjl)).

#### Lightweight SR

*Training*: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [DF2K](https://openmmlab.medium.com/awesome-datasets-for-super-resolution-introduction-and-pre-processing-55f8501f8b18).

*Testing*: Set5, Set14, BSD100, Urban100, Manga109 ([Google Drive](https://drive.google.com/file/d/1SbdbpUZwWYDIEhvxQQaRsokySkcYJ8dq/view?usp=sharing)/[Baidu Netdisk](https://pan.baidu.com/s/1zfmkFK3liwNpW4NtPnWbrw?pwd=nbjl)).

*Preparing*: Please refer to the [Dataset Preparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) of BasicSR.

#### Omni-Directional-Image SR

*Training*: [lau dataset](https://drive.google.com/file/d/1FjEzVh7-0swloClKCVnctUS8Wmlz3Ibv/view?usp=drive_link).

*Preparing*: Please refer to the [Step 1&2&3](https://github.com/Fanghua-Yu/OSRT?tab=readme-ov-file#data-preparation) of OSRT.

▶️ How to Test
---
Clone this repository and change the directory to `./codes`
```
git clone https://github.com/icandle/CAMixerSR.git
cd codes
```
#### Large-Image SR
*Testing*: Change the dataset path of [example option](https://github.com/icandle/CAMixerSR/blob/main/codes/options/test/test_2K.yml) to your datasets and test with the command:
```
# 2K
python basicsr/test.py -opt ../options/test/test_2K.yml
# 4K/8K
python basicsr/test.py -opt ../options/test/test_8K.yml
```
*Note*: We use [TileModel](https://github.com/icandle/CAMixerSR/blob/main/codes/basicsr/models/Tile_model.py) with *Tile* 64x64 and *Overlap* 4 to constrain the calculations.

#### Lightweight SR
*Testing*: Change the dataset path of [example option](https://github.com/icandle/CAMixerSR/blob/main/codes/options/test/test_x4.yml) to your datasets and test with the command:
```
# x2
python basicsr/test.py -opt ../options/test/test_x2.yml
# x4
python basicsr/test.py -opt ../options/test/test_x4.yml
```

✨ Results
---
TBD


💖 Acknowledgments
---
We would thank [BasicSR](https://github.com/XPixelGroup/BasicSR), [ClassSR](https://github.com/XPixelGroup/ClassSR), and [OSRT](https://github.com/Fanghua-Yu/OSRT) for their enlightening work!

🎓 Citation
---
```
@article{wang2024camixersr,
  title={CAMixerSR: Only Details Need More ``Attention"},
  author={Wang, Yan and Liu, Yi and Zhao, Shijie and Li, Junlin and Zhang, Li},
  journal={arXiv preprint arXiv:2402.19289},
  year={2024}
}
```

