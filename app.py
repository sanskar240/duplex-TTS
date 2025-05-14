import gradio as gr
import soundfile as sf
from llm.local_llm import get_response
from tts.vits_infer import load_vits_model, infer_waveform

# Load VITS model
model_path = "checkpoints/vits/model.pth"
config_path = "checkpoints/vits/config.yaml"
model, hps = load_vits_model(model_path, config_path)

def chat_tts(prompt, pitch_scale):
    response = get_response(prompt)
    audio, sr = infer_waveform(response, model, hps, length_scale=pitch_scale)
    sf.write("assets/output.wav", audio, sr)
    return response, "assets/output.wav"

gr.Interface(
    fn=chat_tts,
    inputs=[
        gr.Textbox(label="Your Message"),
        gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Pitch Control")
    ],
    outputs=[
        gr.Textbox(label="EchoBot Response"),
        gr.Audio(label="Voice Output", type="filepath")
    ],
    title="EchoBot — Local LLM + VITS",
    description="Chat with a local language model and hear it respond using VITS."
).launch()
