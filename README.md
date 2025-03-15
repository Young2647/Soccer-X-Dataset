<h1 align="center">Mojito: LLM-Aided Motion Instructor with Jitter-Reduced Inertial Tokens</h1>

<p align="center">
  <a href="https://cunkaixin.netlify.app" target="_blank">Ziwei Shan</a><sup>1,*</sup>,
  <a href="https://tropinoneh.github.io/profile/" target="_blank">Yaoyu He</a><sup>1,*</sup>,
  <a href="https://afterjourney00.github.io/" target="_blank">Chengfeng Zhao</a><sup>1,*,&dagger;</sup>,
  <a href="https://alt-js.github.io/" target="_blank">Jiashen Du</a><sup>1</sup>,
  <br>
  <a href="https://zhanglele12138.github.io/" target="_blank">Jingyan Zhang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=YvwsqvYAAAAJ&hl=en" target="_blank">Qixuan Zhang</a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=R9L_AfQAAAAJ&hl=en" target="_blank">Jingyi Yu</a><sup>1,&Dagger;</sup>,
  <a href="https://www.xu-lan.com/" target="_blank">Lan Xu</a><sup>1,&Dagger;</sup>
</p>
<p align="center">
  <sup>1</sup>ShanghaiTech University&nbsp;&nbsp;
  <sup>2</sup>Deemos Technology
  <br>
  <i><sup>*</sup>Equal contribution</i>
  <br>
  <i><sup>&dagger;</sup>Project lead</i><i> &nbsp;&nbsp; <sup>&Dagger;</sup>Corresponding author</i>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://koyui.github.io/mojito/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>
<div align="center">
  <img width="900px" src="./assets/teaser.png"/>
</div>

## TODO
- [ ] Release code and pretrained model.

## 🚀 Getting Started

### 1. Environment Setup

We tested our environment on `Ubuntu 20.04 LTS` and `Windows 11` with `CUDA 12.1`.

```bash
conda create python=3.10 --name simitor
conda activate simitor

conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# ignore deepspeed installation if using Win 11
DS_BUILD_OPS=1 DS_BUILD_CUTLASS_OPS=0 DS_BUILD_RAGGED_DEVICE_OPS=0 DS_BUILD_EVOFORMER_ATTN=0 pip install deepspeed

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install "fastapi[standard]"
```

### 2. Prepare Body Model and Weights

Download [SMPL-H](https://mano.is.tue.mpg.de/login.php) (the extended SMPL+H model) and put the models under `body_model/` folder. The structure of `body_model/` folder should be:

```
body_model/
|--body_model.py
|--utils.py
|--smplh/
|----info.txt
|----LICENSE.txt
|----female/
|------model.npz
|----male/
|------model.npz
|----neutral/
|------model.npz
```

## 🏄‍♂️ Contributors

- Ziwei Shan - [koyui](https://github.com/koyui)
- Yaoyu He - [TropinoneH](https://github.com/TropinoneH)
- Chengfeng Zhao - [AfterJourney00](https://github.com/AfterJourney00)
- Jiashen Du - [ALT-JS](https://github.com/ALT-JS)

<!-- ## 📖 Citation -->
## 📖 Citation
If you find our code or paper helps, please consider citing:
```bibtex
@article{shan2025mojito,
  title   = {Mojito: LLM-Aided Motion Instructor with Jitter-Reduced Inertial Tokens},
  author  = {Shan, Ziwei and He, Yaoyu and Du, Jiashen and Zhao, Chengfeng and Zhang, Jingyan and 
             Zhang, Qixuan and Yu, Jingyi and Xu, Lan},
  journal = {arXiv preprint arXiv:},
  year    = {2025}
}
```

## Acknowledgments

Thanks to the following work that we refer to and benefit from:
- [MotionGPT](https://github.com/OpenMotionLab/MotionGPT): the overall framework;
- [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct): the causal language model;
- [EgoEgo](https://github.com/lijiaman/egoego_release): the SMPL-H body model script;
- [TransPose](https://github.com/Xinyu-Yi/TransPose): the data pre-processing of TotalCapture dataset;
- [SmoothNet](https://github.com/cure-lab/SmoothNet): SMPL pose smoother

## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.