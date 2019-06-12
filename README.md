# VPoser: Variational Human Pose Prior
![alt text](github_data/vposer_samples.png "Novel Human Poses Sampled From the VPoser.")
## Description
Human joint configuration, also called as pose, is restricted by biomechanics of our body. 
Utilizing these constrains accuratly would be a corner stone of many computer vision tasks, 
such as estimating 3D human body parameters from 2D keypoints, detecting anomolies, and etc.

Here we present a method that is used in [SMPLify-X](https://smpl-x.is.tue.mpg.de/). 
Our variational human pose prior, named as VPoser, has the following features: 
 - is end-to-end differentiable
 - provides a way to penalizes impossible poses while allowing possible ones
 - effectively considers interdependency of configurations of the joints
 - intorduces a more efficient, and lower dimensional representation for human pose
 - can be used as a generative source for data dependent tasks

## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Loading trained models](#loading-trained-models) 
  * [Train VPoser](#train-vposer)
  * [Tutorials](#tutorials)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)

## Installation

To install the model simply you can:
1. To install from PyPi simply run: 
  ```bash
  pip install human_body_prior
  ```
2. Clone this repository and install it using the *setup.py* script: 
```bash
git clone https://github.com/nghorbani/human_body_prior
python setup.py install
```

## Loading Trained Models

To download the *VPoser* trained models go to the [SMPL-X project website](https://smpl-x.is.tue.mpg.de/) and register to get access to the downloads section. Afterwards, you can follow the [model loading tutorial](human_body_prior/tutorials/README.md) to load and use your trained VPoser models.

## Train VPoser
We train VPoser, using a [variational autoencoder](https://arxiv.org/abs/1312.6114), 
which learns a latent representation of human pose and regularizes the distribution of the latent code to be a normal distribution.
We train our prior on the data released by [AMASS](https://amass.is.tue.mpg.de/), 
namely SMPL pose parameters of various publicly available human motion capture datasets. 
You can follow the [data preparation tutorial](human_body_prior/data/README.md) to learn how to download and prepare AMASS for VPoser.
Afterwards, you can [train VPoser from scratch](human_body_prior/train/README.md). 

## Tutorials
* [VPoser PoZ Space for Body Models](human_body_prior/body_model/README.md)
* [Sampling Novel Body Poses from VPoser](human_body_prior/tutorials/README.md)
* [Preparing VPoser Training Dataset](human_body_prior/data/README.md)
* [Train VPoser from Scratch](human_body_prior/train/README.md)

## Citation
Please cite the following paper if you use this code directly or indirectly in your research/projects.
```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
Also note that if you consider training your own VPoser for your research using AMASS dataset, 
then please follow its respective citation guideline.
 
## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Contact

If you have any questions you can contact us at [smplx@tuebingen.mpg.de](mailto:smplx@tuebingen.mpg.de).


## Acknowledgments
We thank authors of [AMASS](https://amass.is.tue.mpg.de/) for their early release of their data to us for this project.
We thank [Partha Ghosh](https://ps.is.tuebingen.mpg.de/person/pghosh) for the helpfull disscussions and insights that helped with this project.