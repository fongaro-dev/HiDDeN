# HiDDeN: Distributed Multi-GPU Tensorflow Implementation

Distributed Multi-GPU tensorflow implementation the of paper ["HiDDeN: Hiding Data With Deep Networks" by Jiren Zhu*, Russell Kaplan*, Justin Johnson, and Li Fei-Fei](https://arxiv.org/abs/1807.09937).
*: These authors contributed equally

While the authors of this extremely influential paper already distributed a [Lua+Torch implementation](https://github.com/jirenz/HiDDeN), Tensorflow is arguably a more widely used framework and I wanted to strengthen my expertise in it, in particular in regards to steganography, adversarial generative AI and multi-GPU distributed training.

For ease of experimentation the code is presented in the form of a Jupyter notebook, with a corresponding python script. Further refinement is reccomended for use in production code.

NOTE: This is a work in progress, in particular not all the noise types analyzed in the paper are currently implemented here.

## Requirements

You will require a [Tensorflow](https://www.tensorflow.org/install/pip) and, optionally, [Tensorboard](https://www.tensorflow.org/tensorboard) for visualization purposes. If you plan on using the GPU distributed aspect of this code please ensure your [GPU is correctly configured](https://www.tensorflow.org/guide/gpu). Furterh, (ana)conda envinroments are reccomended for this project.

## Data

While the orginal paper used the [coco dataset](https://cocodataset.org/#download) I chose to use the [stl10 dataset](https://www.tensorflow.org/datasets/catalog/stl10) as it still presents a large amount of images (with over 110K unlabelled) while allowing an easier distribution thanks to the Tensorflow datasets framework.

## Usage

Once the script is launched ./runs and ./ckpt folders will be created containing the Tensorboard and Tensorflow checkpoint data, respectively.