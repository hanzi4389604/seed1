# seed1
# EfficientNet

https://arxiv.org/abs/1905.11946

## Prerequisites

- Ubuntu 20.04
- Python 3.11
  - torch 1.1.0
  - torchvision 0.3.0

## Usage

### Train

```shell
$ python train.py -c ./configs/custom.yaml -r /path to efficientnet.././datasets/combine_of_Ama_Bra_Leg_15spp
```

## TO-DO

- Automate seed classication for updating seed images
- integrate a larger model (transformer?)
- Integrate the videometer-generated ML features to improve classification
- Increase seed image collection repeats to enlarge seed image pool or diversity
- Include TEVO extra 7 spp into the imageset
- Check how increased spp number could improve DL performance
- develop DL-based seed image segmentation

## References

- https://arxiv.org/abs/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
