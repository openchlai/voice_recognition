from transformers import pipeline
import gradio as gr

# change to "your-username/the-name-you-picked"
pipe = pipeline(
    model="openai/whisper-small",
    tokenizer="openai/whisper-small",
    task="automatic-speech-recognition",
    device="cpu"
    )

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
    outputs="text",
    title="Whisper Small Swahili",
    description="Realtime demo for Swahili speech recognition using a fine-tuned Whisper small model.",
)

iface.launch(share=True)
