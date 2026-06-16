import sounddevice as sd
import soundfile as sf
from pathlib import Path
from voice_assistant.utils.logger import logger

def play_audio(wav_path: Path, blocking: bool = True) -> bool:
    """Plays back a WAV audio file on system speakers using sounddevice.
    
    Args:
        wav_path: Absolute path to the WAV file.
        blocking: If True, blocks until playback finishes. Default is True.
        
    Returns:
        True if playback succeeds, False otherwise.
    """
    if not wav_path.exists():
        logger.error(f"Playback failed: Audio file not found at {wav_path}")
        return False
        
    logger.info(f"Playing audio file (blocking={blocking}): {wav_path}")
    
    try:
        data, fs = sf.read(str(wav_path))
        sd.play(data, fs)
        if blocking:
            sd.wait()
        logger.info("Audio playback initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to play audio file: {e}")
        return False

def stop_audio():
    """Instantly halts any active sounddevice audio playback."""
    try:
        sd.stop()
        logger.info("Audio playback stopped/interrupted.")
    except Exception as e:
        logger.error(f"Failed to stop active audio playback: {e}")

