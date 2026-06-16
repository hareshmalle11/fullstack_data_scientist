import os
from pathlib import Path
from voice_assistant.config.settings import Settings
from voice_assistant.utils.logger import logger
from voice_assistant.utils.exceptions import WhisperFailureException

# Cache the loaded model globally in memory
_model_instance = None

def load_model(force_reload: bool = False) -> object:
    """Loads the Faster-Whisper model into memory on CPU only."""
    global _model_instance
    if _model_instance is not None and not force_reload:
        return _model_instance

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        logger.error(f"Failed to import faster-whisper package: {e}")
        raise WhisperFailureException("faster-whisper is not installed. Please check requirements.txt.") from e

    model_size = Settings.WHISPER_MODEL
    
    # Check if model has been downloaded/cached locally yet
    try:
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_size}"
        if not hf_cache_dir.exists():
            msg = (
                f"Whisper model '{model_size}' is not cached locally. "
                f"Downloading (~460MB for 'small') from Hugging Face Hub... "
                f"This only happens once. Please wait..."
            )
            logger.info(msg)
            print(f"\n📥 {msg}\n")
    except Exception:
        pass

    try:
        logger.info("Loading Whisper model on CPU (int8)...")
        # int8 quantization is fast and light for CPU inference
        _model_instance = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8"
        )
        logger.info("Whisper model loaded successfully on CPU.")
        return _model_instance
    except Exception as e:
        logger.error(f"Failed to load Whisper model on CPU: {e}")
        raise WhisperFailureException(f"Error initializing Whisper engine: {e}") from e

def transcribe_audio(audio_path: Path) -> str:
    """Transcribes an input WAV audio file into plain text on CPU.
    
    Args:
        audio_path: Absolute path to the WAV recording file.
        
    Returns:
        Cleaned, stripped transcription text string.
    """
    if not audio_path.exists():
        logger.error(f"Transcription failed: Audio file not found at {audio_path}")
        return ""
        
    try:
        model = load_model()
        logger.info(f"Running Whisper transcription on file: {audio_path}")
        
        # Transcribe audio file
        segments, info = model.transcribe(
            str(audio_path), 
            beam_size=5,
            language="en"  # Standardize on English for reliability
        )
        
        # Merge chunks
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)
            
        transcription = "".join(text_segments).strip()
        logger.info(f"Transcription completed. Text: '{transcription}'")
        return transcription
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise WhisperFailureException(f"Failed to transcribe audio file: {e}") from e
