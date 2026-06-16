import numpy as np
from voice_assistant.utils.logger import logger

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalizes the audio buffer to have a max amplitude close to 1.0 (or max int16)."""
    if audio_data.size == 0:
        return audio_data
    
    # If integer PCM, convert to float for normalization
    is_int = np.issubdtype(audio_data.dtype, np.integer)
    if is_int:
        float_data = audio_data.astype(np.float32) / 32768.0
    else:
        float_data = audio_data
        
    max_val = np.max(np.abs(float_data))
    if max_val > 0:
        normalized = float_data / max_val
    else:
        normalized = float_data
        
    if is_int:
        # Scale back to int16 range
        return (normalized * 32767.0).astype(np.int16)
    return normalized

def is_silent(audio_data: np.ndarray, threshold: float = 0.002) -> bool:
    """Detects if the recorded audio buffer is essentially silent.
    
    Args:
        audio_data: Numpy array of audio data
        threshold: Amplitude threshold below which it's considered silent
    """
    if audio_data.size == 0:
        return True
        
    # Convert to float for absolute comparison
    if np.issubdtype(audio_data.dtype, np.integer):
        abs_data = np.abs(audio_data.astype(np.float32) / 32768.0)
    else:
        abs_data = np.abs(audio_data)
        
    mean_amplitude = np.mean(abs_data)
    logger.info(f"Audio check - Mean amplitude: {mean_amplitude:.5f} (Threshold: {threshold})")
    return mean_amplitude < threshold
