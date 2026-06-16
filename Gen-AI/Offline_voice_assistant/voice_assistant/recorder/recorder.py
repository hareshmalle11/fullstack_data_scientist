import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
from pathlib import Path
from voice_assistant.config.settings import Settings
from voice_assistant.recorder.audio_utils import normalize_audio, is_silent
from voice_assistant.utils.logger import logger
from voice_assistant.utils.exceptions import MicrophoneMissingException

def record_audio(
    output_path: Path, 
    sample_rate: int = Settings.SAMPLE_RATE, 
    channels: int = Settings.CHANNELS
) -> bool:
    """Records audio from the system microphone using Enter keypress controls.
    
    Args:
        output_path: Target path to write the WAV audio recording
        sample_rate: Recording sample rate (default 16000)
        channels: Number of input channels (default 1)
        
    Returns:
        True if successfully recorded and saved a non-silent file, False otherwise.
    """
    # Verify input microphone exists
    try:
        sd.query_devices(kind='input')
    except Exception as e:
        logger.error(f"Microphone device query failed: {e}")
        raise MicrophoneMissingException("No active input microphone device detected on this system.")

    audio_chunks = []
    
    def callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Sounddevice warning callback status: {status}")
        audio_chunks.append(indata.copy())

    input("\n🎤 Press Enter to START recording...")
    logger.info("Recording started.")
    
    try:
        # Open the stream in int16 PCM format
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='int16',
            callback=callback
        )
        with stream:
            input("🎙️ Recording... Press Enter again to STOP recording.")
    except Exception as e:
        logger.error(f"Failed to open audio input stream: {e}")
        raise MicrophoneMissingException(f"Error opening microphone stream: {e}")
        
    logger.info("Recording stopped.")
    
    if not audio_chunks:
        logger.warning("No audio data was captured during recording.")
        return False
        
    # Concatenate all numpy array blocks
    raw_audio = np.concatenate(audio_chunks, axis=0)
    
    # Check if empty or silent
    if is_silent(raw_audio):
        logger.warning("Captured recording was silent. Skipping save.")
        print("⚠️ Recording was silent. Please speak into your microphone.")
        return False

    # Normalize audio file
    normalized_audio = normalize_audio(raw_audio)
    
    # Ensure folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as standard WAV
    try:
        wav.write(str(output_path), sample_rate, normalized_audio)
        logger.info(f"Saved recording to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write audio WAV file: {e}")
        return False

def record_fixed_duration(
    output_path: Path,
    duration: float = 5.0,
    sample_rate: int = Settings.SAMPLE_RATE,
    channels: int = Settings.CHANNELS
) -> bool:
    """Records audio from system microphone for a fixed duration.
    
    Args:
        output_path: Target path to write the WAV audio recording
        duration: Time to record in seconds (default 5.0)
        sample_rate: Recording sample rate (default 16000)
        channels: Number of input channels (default 1)
        
    Returns:
        True if successfully recorded and saved a non-silent file, False otherwise.
    """
    try:
        sd.query_devices(kind='input')
    except Exception as e:
        logger.error(f"Microphone device query failed: {e}")
        raise MicrophoneMissingException("No active input microphone device detected on this system.")

    logger.info(f"Recording for a fixed duration of {duration} seconds...")
    try:
        # Record audio synchronously using sounddevice rec
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype='int16'
        )
        sd.wait()  # Wait for the recording to finish
    except Exception as e:
        logger.error(f"Failed to run fixed-duration recording stream: {e}")
        raise MicrophoneMissingException(f"Error during microphone recording: {e}")

    logger.info("Recording complete. Processing audio...")
    if recording.size == 0 or is_silent(recording):
        logger.warning("Captured recording was silent.")
        return False

    normalized_audio = normalize_audio(recording)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        wav.write(str(output_path), sample_rate, normalized_audio)
        logger.info(f"Saved fixed-duration recording to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write audio WAV file: {e}")
        return False

def record_vad(
    output_path: Path,
    silence_seconds: float = 1.5,
    timeout: float = 15.0,
    sample_rate: int = Settings.SAMPLE_RATE,
    channels: int = Settings.CHANNELS
) -> bool:
    """Records audio from system microphone, automatically stopping on silence (VAD).
    
    Args:
        output_path: Target path to write the WAV audio recording
        silence_seconds: Number of silent seconds to trigger stop (default 1.5)
        timeout: Maximum duration to record in seconds (default 15.0)
        sample_rate: Recording sample rate (default 16000)
        channels: Number of input channels (default 1)
        
    Returns:
        True if successfully recorded and saved a non-silent file, False otherwise.
    """
    try:
        sd.query_devices(kind='input')
    except Exception as e:
        logger.error(f"Microphone device query failed: {e}")
        raise MicrophoneMissingException("No active input microphone device detected on this system.")

    audio_chunks = []
    
    def callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Sounddevice callback warning: {status}")
        audio_chunks.append(indata.copy())

    # VAD State parameters
    speech_threshold = 0.025
    silence_threshold = 0.015
    
    start_time = time.time()
    last_speech_time = time.time()
    speech_started = False
    silence_detected = False
    
    logger.info("VAD: Starting non-blocking input stream...")
    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='int16',
            callback=callback
        )
        
        with stream:
            while time.time() - start_time < timeout:
                time.sleep(0.1)
                
                if audio_chunks:
                    # Analyze latest audio block volume
                    latest_block = audio_chunks[-1]
                    volume = np.mean(np.abs(latest_block.astype(np.float32) / 32768.0))
                    
                    if volume > speech_threshold:
                        if not speech_started:
                            logger.info("VAD: User started speaking.")
                            speech_started = True
                        last_speech_time = time.time()
                    
                    if speech_started:
                        if volume < silence_threshold:
                            elapsed_silence = time.time() - last_speech_time
                            if elapsed_silence > silence_seconds:
                                logger.info(f"VAD: Silence threshold met ({elapsed_silence:.2f}s). Stopping stream.")
                                silence_detected = True
                                break
                                
                # Exit if user never started speaking in the first 4 seconds
                if not speech_started and (time.time() - start_time) > 4.0:
                    logger.info("VAD: Timeout waiting for user speech. Exiting.")
                    break
                    
    except Exception as e:
        logger.error(f"VAD: Stream execution error: {e}")
        raise MicrophoneMissingException(f"Error during VAD stream capture: {e}")

    if not audio_chunks or not speech_started or (not silence_detected and not (time.time() - start_time >= timeout)):
        logger.warning("VAD: No speech captured or recording was aborted.")
        return False

    raw_audio = np.concatenate(audio_chunks, axis=0)

    normalized_audio = normalize_audio(raw_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        wav.write(str(output_path), sample_rate, normalized_audio)
        logger.info(f"VAD: Saved audio recording to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"VAD: Failed to write audio WAV file: {e}")
        return False


