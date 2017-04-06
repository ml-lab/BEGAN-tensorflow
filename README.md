# BEGAN in Tensorflow

Tensorflow implementation of [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717).

![alt tag](./assets/model.png)


## Requirements

- Python 2.7
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [tqdm](https://github.com/tqdm/tqdm)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [requests](https://github.com/kennethreitz/requests) (Only used for downloading CelebA dataset)


## Usage

First download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets with:

    $ apt-get install p7zip-full # ubuntu
    $ brew install p7zip # Mac
    $ python download.py

or you can use your own dataset by placing images like:

    data
    └── YOUR_DATASET_NAME
        ├── xxx.jpg (name doesn't matter)
        ├── yyy.jpg
        └── ...

To train a model:

    $ python main.py --dataset=CelebA --num_gpu=1
    $ python main.py --dataset=YOUR_DATASET_NAME --num_gpu=4

To test a model (use your `load_path`):

    $ python main.py --dataset=CelebA --load_path=./logs/CelebA_0405_124806 --num_gpu=0 --is_train=False --split valid


## Results

- [BEGAN-tensorflow](https://github.com/carpedm20/began-tensorflow) at least can generate human faces but [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch) can't
- Both [BEGAN-tensorflow](https://github.com/carpedm20/began-tensorflow) and [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch) shows "modal collapses" and I guess this is due to a wrong scheuduling of lr. Paper mentioned that "simply reducing the lr was sufficient to avoid them" but the scheuduling that I've used doesn't reduce it's lr until 100k step.


(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
