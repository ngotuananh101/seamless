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
            config = SeamlessM4TConfig(
                vocab_size=256102,
                t2u_vocab_size=10082,
                hidden_size=1024,
                initializer_range=0.02,
                layer_norm_eps=1e-05,
                use_cache=True,
                max_position_embeddings=1024,
                is_encoder_decoder=True,
                encoder_layerdrop=0.05,
                decoder_layerdrop=0.05,
                activation_function='relu',
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                scale_embedding=True,
                encoder_layers=24,
                encoder_ffn_dim=8192,
                encoder_attention_heads=16,
                decoder_layers=24,
                decoder_ffn_dim=8192,
                decoder_attention_heads=16,
                decoder_start_token_id=3,
                max_new_tokens=256,
                pad_token_id=0,
                bos_token_id=2,
                eos_token_id=3,
                speech_encoder_layers=24,
                speech_encoder_attention_heads=16,
                speech_encoder_intermediate_size=4096,
                speech_encoder_hidden_act='swish',
                speech_encoder_dropout=0.0,
                add_adapter=True,
                speech_encoder_layerdrop=0.1,
                feature_projection_input_dim=160,
                num_conv_pos_embeddings=128,
                num_conv_pos_embedding_groups=16,
                adaptor_kernel_size=8,
                adaptor_stride=8,
                adaptor_dropout=0.1,
                num_adapter_layers=1,
                position_embeddings_type='relative',
                rotary_embedding_base=10000,
                max_source_positions=4096,
                conv_depthwise_kernel_size=31,
                t2u_bos_token_id=0,
                t2u_pad_token_id=1,
                t2u_eos_token_id=2,
                t2u_decoder_start_token_id=2,
                t2u_max_new_tokens=1024,
                t2u_encoder_layers=6,
                t2u_encoder_ffn_dim=8192,
                t2u_encoder_attention_heads=16,
                t2u_decoder_layers=6,
                t2u_decoder_ffn_dim=8192,
                t2u_decoder_attention_heads=16,
                t2u_max_position_embeddings=2048,
                sampling_rate=16000,
                upsample_initial_channel=512,
                upsample_rates=[5, 4, 4, 2, 2],
                upsample_kernel_sizes=[11, 8, 8, 4, 4],
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                leaky_relu_slope=0.1,
                unit_hifi_gan_vocab_size=10000,
                unit_embed_dim=1280,
                lang_embed_dim=256,
                spkr_embed_dim=256,
                vocoder_num_langs=36,
                vocoder_num_spkrs=200,
                variance_predictor_kernel_size=3,
                var_pred_dropout=0.5,
                vocoder_offset=4
            )
            model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-medium", config=config)
            model.to(device)
            processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-medium")
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
        output = model.generate(**text_inputs, tgt_lang=tgt_lang)
        
        # Decode text
        translated_text = processor.decode(output.sequences[0], skip_special_tokens=True)
        
        # Extract audio
        if output.waveform is not None:
             audio_array = output.waveform.cpu().detach().squeeze().numpy()
             sample_rate = model.config.sampling_rate
             return (sample_rate, audio_array), translated_text, "Translation complete."
        else:
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
        
        output = model.generate(**audio_inputs, tgt_lang=tgt_lang)
        
        # Decode text
        translated_text = processor.decode(output.sequences[0], skip_special_tokens=True)
        
        if output.waveform is not None:
             audio_array = output.waveform.cpu().detach().squeeze().numpy()
             sample_rate = model.config.sampling_rate
             return (sample_rate, audio_array), translated_text, "Translation complete."
        else:
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
