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
   ```
2. Locate the `inference.py` file in the cloned repository (`maha_tts` folder) and replace it with the `inference.py` provided in this repository.

This replacement ensures that a new folder named `mel_spectrograms` is created during inference. The folder will contain:
- `.npy` files for mel-spectrogram data.
- Corresponding mel-spectrogram images.

These files are essential for running inferences with different vocoders.

---

### 2. Run Inference with Different Vocoders

#### Using **MelGAN**
To generate audio using MelGAN:
```bash
python MelGAN.py --input_folder <path_to_mel_spectrograms_folder> --output_folder <path_to_output_folder>
```

#### Using **WaveGlow**
To generate audio using WaveGlow:
```bash
python WaveGlow.py --input_folder <path_to_mel_spectrograms_folder> --output_folder <path_to_output_folder>
```

---

## Dependencies

Ensure the following dependencies are installed:
- `soundfile==0.12.1`
- `numpy==1.26.2`
- `torch==2.4.1+cu118`

You can install them using:
```bash
pip install soundfile==0.12.1 numpy==1.26.2 torch==2.4.1+cu118
```

---

## Known Issues

- **Hindi Language Compatibility:**  
  Both MelGAN and WaveGlow encounter issues when processing some Hindi mel-spectrogram `.npy` files, resulting in incomplete or failed audio generation. Further debugging and tuning may be required to resolve this.

---

This repository serves as a foundation for exploring vocoder performance and improving the MahaTTS pipeline. Contributions and suggestions are welcome!
