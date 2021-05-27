# If Monet Loved Dogs More

## Training Data

- All of Monet's paintings parsed from wikiart.org
- Stanford dog [dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) from Kaggle

Details

- for the dog dataset, images that meet both of the following conditions are used during training
  - object size to image size ratio greater than 0.8
  - both image hight and width are greater than 256
  - however, while training, images with one side greater than 512 are likely to cause memory issues.
  - therefore, images are resized to be less than or equal to 512x512 in the data generator
  - in the end 1362 pairs of images are used in the training with monet's paintings being the bottleneck

## How to use

```bash
# build the container
docker build -f Dockerfile.train -t monet_cyclegan .

# start the container
docker run --gpus all -v $(pwd):/work -p 8888:8888 -p 6006:6006 -it monet_cyclegan

# start jupyter notebook
jupyter notebook --ip=0.0.0.0 --allow-root

# see details in the notebook folder
```

## Implementations

- [x] an image buffer that stores previously generated images. this is used to update discriminators using a history of generated images
- [x] linearly decreases the learning rate to 0 only after the first 100 epochs
- [x] BCE and MSE loss
- [x] U-Net and ResNet generator

## Training Details

<img src="https://github.com/yueying-teng/cycleGAN_if_monet_loved_dogs_more/blob/master/tf_board/monet_cycle_loss.png" height="300">

<img src="https://github.com/yueying-teng/cycleGAN_if_monet_loved_dogs_more/blob/master/tf_board/monet_identity_loss.png" height="300">

<img src="https://github.com/yueying-teng/cycleGAN_if_monet_loved_dogs_more/blob/master/tf_board/monet_gen_loss.png" height="300">

<img src="https://github.com/yueying-teng/cycleGAN_if_monet_loved_dogs_more/blob/master/tf_board/monet_disc_loss.png" height="300">

- ResNet based generator and 3-layer PatchGAN discriminator
- LS GAN loss
- image buffer with pool size: 50
- learning rate decay (linearly towards 0 after the first 100 epochs)
- 200 epochs with batch size 1

## Gallery

click [here]()
