import torch
from tacotron2.model import Tacotron2
from waveglow.denoiser import Denoiser
from waveglow.glow import WaveGlow
from transformers import GPT2Tokenizer

# Function to load Tacotron 2 model
def load_tacotron2(model_path):
    tacotron2 = Tacotron2()
    tacotron2.load_state_dict(torch.load(model_path)['state_dict'])
    tacotron2.eval().cuda()
    return tacotron2

# Function to load WaveGlow model
def load_waveglow(model_path):
    waveglow = torch.load(model_path)['model']
    waveglow.cuda().eval()
    return waveglow

# Text-to-Mel Spectrogram
def text_to_spectrogram(tacotron2, input_text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer.encode(input_text, return_tensors='pt').cuda()
    
    with torch.no_grad():
        mel_outputs, _, _, _ = tacotron2.infer(inputs)
    return mel_outputs

# Convert Mel Spectrogram to Audio using WaveGlow
def mel_to_audio(waveglow, mel_spec):
    with torch.no_grad():
        audio = waveglow.infer(mel_spec)
    return audio

# Main function to generate speech
def generate_speech(input_text, tacotron2_model_path, waveglow_model_path):
    # Load models
    tacotron2 = load_tacotron2(tacotron2_model_path)
    waveglow = load_waveglow(waveglow_model_path)

    # Generate Mel-spectrogram from text
    mel_spec = text_to_spectrogram(tacotron2, input_text)

    # Generate audio from Mel-spectrogram
    audio = mel_to_audio(waveglow, mel_spec)
    
    # Optionally, apply denoiser to clean up the audio
    denoiser = Denoiser(waveglow)
    audio_denoised = denoiser(audio, strength=0.01)
    
    # Save audio
    audio_output = audio_denoised.cpu().numpy().astype('float32')
    output_wav_path = 'output/generated_audio.wav'
    save_wav(audio_output, output_wav_path)
    print(f"Audio saved at: {output_wav_path}")

# Helper function to save audio
def save_wav(audio, file_path, sample_rate=22050):
    from scipy.io.wavfile import write
    write(file_path, sample_rate, audio)

if __name__ == "__main__":
    # Paths to models
    tacotron2_model_path = 'models/tacotron2_statedict.pt'
    waveglow_model_path = 'models/waveglow_256channels.pt'

    # Input text (could be generated from GPT-2 or provided manually)
    input_text = "These are the generated lyrics for the song."

    # Generate speech
    generate_speech(input_text, tacotron2_model_path, waveglow_model_path)
