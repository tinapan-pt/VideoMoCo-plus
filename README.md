# Contrastive Learning of Video Representations with Temporally Adversarial Examples
Pytroch implementation of our paper "Contrastive Learning of Video Representations with Temporally Adversarial Examples", a journal extension of our preliminary work presented in [CVPR 2021](https://arxiv.org/abs/2103.05905). Extensive additional ananlysis are presented in this version.

The Pytorch implementation of our previous CVPR 2021 work is available at: https://github.com/tinapan-pt/VideoMoCo.

# Overview
Framework of the proposed approach.

We introduce generative adversarial learning to improve the temporal robustness of the encoder. We use a generator to temporally drop out several frames from this sample. The discriminator is then learned to encode similar feature representations regardless of frame removals. By adaptively dropping out different frames during training iterations of adversarial learning, we augment this input sample to train a temporally robust encoder. Second, we propose a temporally adversarial decay to model key attenuation in the memory queue when computing the contrastive loss.
![pipeline.png](https://i.loli.net/2021/05/08/ZVGAwKk2mIuY1aP.png)

# Requirements
- pytroch >= 1.3.0
- tensorboard
- cv2
- kornia

# Usage

## Data preparation

K400 dataset
- Download the K400 dataset from the [official website](https://deepmind.com/research/open-source/kinetics).

## Train
```python
python train.py \  
  --log_dir ./logs_moco \  
  --ckp_dir ./checkpoints_moco \
  -a r2plusd_18 \
  --lr 0.04 \
  -fpc 32 \
  -b 256 \
  -j 128 \
  --epochs 200 \
  --schedule 120 160 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  DATA_DIR/kinetics-400
```
## TODO
Downstream task evaluation

- [ ] Action Recognition
- [ ] Video Retrieval
- [ ] Feature Separation
