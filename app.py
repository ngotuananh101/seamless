import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4TConfig
from languages import (
    text_source_languages,
    speech_target_languages,
    text_source_codes,
    speech_target_codes,
    get_language_name
)

# Global variables to hold model and processor
model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model, processor, device
    if model is None:
        print(f"Loading model on {device}...")
        try:
            model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
            model.to(device)
            processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    return model, processor

def preprocess_audio(audio):
    """
    Convert Gradio audio (sample_rate, numpy_array) to torch tensorresampled to 16kHz.
    """
    if audio is None:
        return None
    
    orig_freq, audio_numpy = audio
    
    # Convert numpy to tensor
    audio_tensor = torch.from_numpy(audio_numpy).float()
    
    # Handle dimensions (Mono/Stereo)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0) # [1, samples]
    elif audio_tensor.dim() == 2:
        # Check orientation
        if audio_tensor.shape[0] == 2: # [channels, samples]
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        else: # [samples, channels] usually
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True).T
            
    # Resample to 16000 Hz (required by Seamless)
    if orig_freq != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=orig_freq, new_freq=16000)
        
    return audio_tensor

def seamless_translate(text, src_lang_name, tgt_lang_name):
    if not text:
        return None, None, "Please enter text."
        
    global model, processor, device
    load_model()
    
    try:
        src_lang = text_source_codes[text_source_languages.index(src_lang_name)]
        tgt_lang = speech_target_codes[speech_target_languages.index(tgt_lang_name)]
        
        print(f"Translating text: '{text}' from {src_lang} to {tgt_lang}")

        text_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(device)
        
        # Generate Audio and Text
        # First generate text
        text_output = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
        # text_output is a tensor of shape [batch, seq_len], decode the first sequence
        translated_text = processor.batch_decode(text_output, skip_special_tokens=True)[0]
        
        # Then generate speech
        audio_output = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=True)
        
        # audio_output is (waveform, waveform_lengths) when generate_speech=True
        if audio_output is not None and len(audio_output) > 0 and audio_output[0] is not None:
             audio_array = audio_output[0].cpu().detach().squeeze().numpy()
             sample_rate = model.config.sampling_rate
             return (sample_rate, audio_array), translated_text, "Translation complete."
        else:
            print(f"DEBUG: No waveform generated. Output: {audio_output}")
            return None, translated_text, "No audio generated."
            
    except Exception as e:
        return None, None, str(e)

def seamless_audio_translate(audio, tgt_lang_name):
    if audio is None:
        return None, None, "Please provide audio input."
        
    global model, processor, device
    load_model()
    
    try:
        tgt_lang = speech_target_codes[speech_target_languages.index(tgt_lang_name)]
        
        audio_tensor = preprocess_audio(audio)
        audio_inputs = processor(audios=audio_tensor, return_tensors="pt").to(device)
        
        print(f"Translating audio to {tgt_lang}")
        
        # First generate text
        text_output = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
        # text_output is a tensor of shape [batch, seq_len], decode the first sequence
        translated_text = processor.batch_decode(text_output, skip_special_tokens=True)[0]
        
        # Then generate speech
        audio_output = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=True)
        
        # audio_output is (waveform, waveform_lengths) when generate_speech=True
        if audio_output is not None and len(audio_output) > 0 and audio_output[0] is not None:
             audio_array = audio_output[0].cpu().detach().squeeze().numpy()
             sample_rate = model.config.sampling_rate
             return (sample_rate, audio_array), translated_text, "Translation complete."
        else:
            print(f"DEBUG: No waveform generated. Output: {audio_output}")
            return None, translated_text, "No audio generated."

    except Exception as e:
        return None, None, str(e)


# Gradio UI
with gr.Blocks(title="Seamless M4T Demo") as demo:
    gr.Markdown("# Facebook Seamless M4T Medium Model Demo")
    gr.Markdown("Translate Text or Audio into high-quality speech and text in various languages.")
    
    with gr.Tabs():
        with gr.Tab("Text to Speech Translation"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label="Input Text", lines=3, placeholder="Enter text to translate...")
                    src_lang_dropdown = gr.Dropdown(choices=text_source_languages, value="Vietnamese", label="Source Language")
                    tgt_lang_dropdown = gr.Dropdown(choices=speech_target_languages, value="English", label="Target Language")
                    text_btn = gr.Button("Translate", variant="primary")
                with gr.Column():
                    output_audio_text = gr.Audio(label="Translated Speech")
                    output_text_text = gr.Textbox(label="Translated Text", lines=3, interactive=False)
                    status_text = gr.Textbox(label="Status/Error", interactive=False)
            
            text_btn.click(seamless_translate, inputs=[input_text, src_lang_dropdown, tgt_lang_dropdown], outputs=[output_audio_text, output_text_text, status_text])

        with gr.Tab("Audio to Speech Translation"):
            with gr.Row():
                with gr.Column():
                    input_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Input Audio")
                    tgt_lang_audio_dropdown = gr.Dropdown(choices=speech_target_languages, value="English", label="Target Language")
                    audio_btn = gr.Button("Translate", variant="primary")
                with gr.Column():
                    output_audio_audio = gr.Audio(label="Translated Speech")
                    output_text_audio = gr.Textbox(label="Translated Text", lines=3, interactive=False)
                    status_audio = gr.Textbox(label="Status/Error", interactive=False)
            
            audio_btn.click(seamless_audio_translate, inputs=[input_audio, tgt_lang_audio_dropdown], outputs=[output_audio_audio, output_text_audio, status_audio])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()
    
    demo.launch(share=args.share)
