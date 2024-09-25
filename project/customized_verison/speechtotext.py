import os
from dotenv import load_dotenv, find_dotenv
import torch
import whisper

# Load environment variables
_ = load_dotenv(find_dotenv())

class SpeechToText:
    def __init__(self, device='auto'):
        self.device = device

    def initialize_model(self):
            if 'STTMODEL' not in globals() or STTMODEL is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
                # Get the model path from environment variables
                model_path = os.getenv("speechtotextmodel_path")
                if not model_path:
                    raise ValueError("The environment variable 'speechtotextmodel_path' is not set.")
                
                # Load the model
                model = whisper.load_model(model_path, device=device).eval()
                
            
                print("STT model initialized.")
            return model

tts_instance = SpeechToText()

global STTMODEL
STTMODEL = tts_instance.initialize_model()
    
def transcribe_audio_from_file(audio_file):
    """
    Transcribe the given audio file using the Whisper model.

    Args:
        audio_file (str): Path to the audio file to be transcribed.

    Returns:
        str: Transcribed text from the audio file.

    Raises:
        ValueError: If the model path is not set in the environment variables.
        RuntimeError: If there is an issue with loading the model or transcribing the audio.
    """
    try:
        # Determine the device to use for inference
         # Transcribe the audio file
        transcription = STTMODEL.transcribe(audio_file, task="transcribe", language="en")
        
        recognized_text = transcription['text']
        print("recognized_text",recognized_text)
        return recognized_text
    
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        raise
