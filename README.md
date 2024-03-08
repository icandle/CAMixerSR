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

This repository contains [PyTorch](https://pytorch.org/) implementation for ***CAMixerSR*** (CVPR 2024).

---
The main codes and pre-trained models have been uploaded. We are planning to replenish the README of CAMixerSR in a few weeks.

---
### Citation
```
@article{wang2024camixersr,
  title={CAMixerSR: Only Details Need More ``Attention"},
  author={Wang, Yan and Zhao, Shijie and Liu, Yi and Li, Junlin and Zhang, Li},
  journal={arXiv preprint arXiv:2402.19289},
  year={2024}
}
```

