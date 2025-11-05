<h1 align="center">SMGDiff: Soccer Motion Generation using diffusion probabilistic models</h1>

<p align="center">
  <a href="https://Young2647.github.io" target="_blank">Hongdi Yang</a><sup>1,*</sup>,
  <a href="https://lichy2004.github.io/homepage/" target="_blank">Chengyang Li</a><sup>1,*</sup>,
  <a href="https://" target="_blank">Zhenxuan Wu</a><sup>1</sup>,
  <a href="https://" target="_blank">Gaozheng Li</a><sup>1</sup>,
  <br>
  <a href="https://" target="_blank">Jingya Wang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=R9L_AfQAAAAJ&hl=en" target="_blank">Jingyi Yu</a><sup>1</sup>,
  <a href="https://" target="_blank">Zhuo Su</a><sup>2</sup>,
  <a href="https://www.xu-lan.com/" target="_blank">Lan Xu</a><sup>1,&dagger;</sup>
</p>
<p align="center">
  <sup>1</sup>ShanghaiTech University&nbsp;&nbsp;
  <sup>2</sup>ByteDance
  <br>
  <i><sup>*</sup>Equal contribution</i> <i>&nbsp;&nbsp; <sup>&dagger;</sup>Corresponding author</i>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2411.16216"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://Young2647.github.io/SMGDiff'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>
<div align="center">
  <img width="900px" src="./assets/teaser.jpg"/>
</div>

## TODO
- [ ] Release code and unity demo.
  - ✅ Release training code.
- ✅ Release whole dataset.

## 📁 Dataset
For datasets, please fill out this [form](./assets/license.pdf) and send an email to apply for license. We will reply in one week. 

## :computer: Instructions
###  1. Set up environment
#### 1.1 Create environment
```shell
conda env create -f smgdiff.yaml
conda activate smgdiff
```
#### 1.2 Install SMPL model
Download [SMPL model](https://smpl.is.tue.mpg.de/download.php)(female/male, 10 shape PCs) and [SMPLify model](https://smplify.is.tue.mpg.de/download.php)(neutral model), and extract folder and put pkls here:
- smpl_models
    - smpl
        - SMPL_FEMALE.pkl
        - SMPL_MALE.pkl
        - SMPL_NEUTRAL.pkl

### 2. Make Soccer Data
After getting the dataset, your dataset folder should be like this
- clip
  - Part1
  - Part2
  - ...
  - Part10
- code
  - ...

run the following code to generate the training data pkl.
```shell
python ./data/make_pose_data_new.py --data_root path/to/your/dataset/clip --export_path ./data/pkls/your_dataset_name.pkl
```

### 3. Run training code
To train your model, run:
```shell
python train.py --name your_train_exp_name --config ./config/soccer_train.json -i ./data/pkls/your_dataset_name.pkl
```

### 4. Trajectory generation Part
Similarly as training the motion diffusion model, first generate the training data pkl:
```shell
cd ./trajectory_generation_part
python ./data/make_data_traj.py --data_root path/to/your/dataset/clip --export_path ./data/pkls/your_dataset_name.pkl
```
Then run the following code to train trajectory generation model:
```shell
cd ./trajectory_generation_part
python ./train.py --name your_train_exp_name --dataset_path ../data/pkls/your_dataset_name.pkl --save_path ./checkpoints
```

<!-- ## 📖 Citation -->
## 📖 Citation
If you find our code or paper helps, please consider citing:
```bibtex
@misc{yang2024smgdiffsoccermotiongeneration,
      title={SMGDiff: Soccer Motion Generation using diffusion probabilistic models}, 
      author={Hongdi Yang and Chengyang Li and Zhenxuan Wu and Gaozheng Li and Jingya Wang and Jingyi Yu and Zhuo Su and Lan Xu},
      year={2024},
      eprint={2411.16216},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16216}, 
}
```

## Acknowledgments

Thanks to the following work that we refer to and benefit from:
- [CAMDM](https://github.com/AIGAnimation/CAMDM): the overall framework;
- [MDM](https://guytevet.github.io/mdm-page/): the diffusion network;

## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.