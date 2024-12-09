import os
import torch
import numpy as np
from scipy.io.wavfile import write
import argparse

def main(input_folder, output_folder):
    # Load the WaveGlow model
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow = waveglow.remove_weightnorm(waveglow)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveglow = waveglow.to(device)
    waveglow.eval()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Sample rate used during training
    sample_rate = 22050

    # Process each .npy file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            mel_path = os.path.join(input_folder, filename)
            
            # Load the mel-spectrogram
            mel = np.load(mel_path)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(device)  # shape (1, 80, T)

            # Generate audio using WaveGlow
            with torch.no_grad():
                audio_tensor = waveglow.infer(mel_tensor)
            audio_numpy = audio_tensor[0].data.cpu().numpy().flatten()  # Flatten the tensor to 1D
            
            # Save the audio to a .wav file
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
            write(output_path, sample_rate, audio_numpy)
            print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio from mel-spectrograms using NVIDIA WaveGlow.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing .npy files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save .wav files")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
