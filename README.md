# camera_noise_modeling

This repository contains the sRGBNoise model forked from [Noise2NoiseFlow](https://github.com/SamsungLabs/Noise2NoiseFlow) modified and pre-trained to generate sensor noise with a distribution similar to the noise of a Basler ace aCA640-750uc color camera.

The added [sRGB_noise_modeling/add_noise.py](sRGB_noise_modeling/add_noise.py) script can augment noise-free RGB images with generated sensor noise using the sRGBNoise model. The model has been trained on images captured by a Basler ace aCA640-750uc color camera. One can use the script as follows:

```
python add_noise.py /input/image.png /output/noisy_image.png
```
