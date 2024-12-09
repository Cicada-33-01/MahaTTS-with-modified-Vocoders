import os
import torch
import numpy as np
import soundfile as sf
import argparse

def main(input_folder, output_folder):
    # Load the vocoder model
    vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
    vocoder.eval()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = vocoder.to(device)

    # Process each .npy file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            mel_path = os.path.join(input_folder, filename)
            
            # Load the mel-spectrogram
            mel = np.load(mel_path)

            # Ensure the mel-spectrogram is in the correct shape for MelGAN: (1, 80, time_steps)
            if len(mel.shape) == 2:
                mel = mel[np.newaxis, :]

            # Convert to a PyTorch tensor and move to device
            mel_tensor = torch.from_numpy(mel).float().to(device)

            # Run inference to generate audio
            with torch.no_grad():
                audio = vocoder.inference(mel_tensor).cpu().numpy()  # Move to CPU and convert to NumPy

            # Save the audio to a .wav file
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
            sf.write(output_path, audio.squeeze(), samplerate=22050)  # Replace with correct sample rate
            print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mel-spectrograms to audio using MelGAN vocoder.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing .npy files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save .wav files")

    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
