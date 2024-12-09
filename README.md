# MahaTTS with Modified Vocoders

This repository aims to enhance the **MahaTTS pipeline** by integrating and testing different vocoders in place of the currently used **HiFi-GAN**. You can find the original MahaTTS repository [here](https://github.com/dubverse-ai/MahaTTS).

## Overview

To facilitate experimentation with different vocoders, we:
1. Manually separate mel-spectrograms and store them as both `.npy` files and images.
2. Provide compatibility for alternative vocoders such as **MelGAN** and **WaveGlow**.

This approach allows for seamless testing of multiple vocoders by reusing the pre-generated mel-spectrograms.

---

## Setup Instructions

### 1. Modify MahaTTS for Mel-Spectrogram Generation
To enable `.npy` and mel-spectrogram image storage:
1. Clone the MahaTTS repository:
   ```bash
   git clone https://github.com/dubverse-ai/MahaTTS.git
