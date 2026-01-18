import os
import torch
from gtts import gTTS
from PIL import Image
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, VideoFileClip
from diffusers import DiffusionPipeline
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer
nltk.download("punkt")

# Define folders
os.makedirs("images", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("clips", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None
).to(device)

# Input story
text_input = input("Enter your story:\n")
sentences = sent_tokenize(text_input)

# Generate images + audio
for i, sentence in enumerate(sentences):
    print(f"Generating image {i+1}...")
    image = pipe(sentence, guidance_scale=3.0, num_inference_steps=15).images[0]
    image_path = f"images/scene_{i+1}.png"
    image.save(image_path)

    print(f"Generating audio {i+1}...")
    tts = gTTS(text=sentence, lang='en')
    audio_path = f"audio/scene_{i+1}.mp3"
    tts.save(audio_path)

# Create video clips
print("Creating video clips...")
for i in range(len(sentences)):
    image_path = f"images/scene_{i+1}.png"
    audio_path = f"audio/scene_{i+1}.mp3"

    audio_clip = AudioFileClip(audio_path)
    img_clip = ImageClip(image_path).set_duration(audio_clip.duration).set_audio(audio_clip).set_fps(24)
    img_clip.write_videofile(f"clips/clip_{i+1}.mp4", fps=24, codec='libx264', audio_codec='aac')

# Combine all clips
print("Combining into final video...")
clips = [VideoFileClip(f"clips/clip_{i+1}.mp4") for i in range(len(sentences))]
final_video = concatenate_videoclips(clips)
final_video.write_videofile("final/final_video.mp4", fps=24, codec='libx264', audio_codec='aac')