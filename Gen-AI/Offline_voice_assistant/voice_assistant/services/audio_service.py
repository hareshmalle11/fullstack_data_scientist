import time
from pathlib import Path
from voice_assistant.config.settings import Settings
from voice_assistant.recorder.recorder import record_audio, record_fixed_duration, record_vad
from voice_assistant.stt.whisper_engine import transcribe_audio
from voice_assistant.tts.piper_engine import text_to_speech
from voice_assistant.tts.speaker import play_audio
from voice_assistant.utils.logger import logger

class AudioService:
    def __init__(self):
        self.input_wav = Settings.RECORDINGS_DIR / "input.wav"
        self.output_wav = Settings.GENERATED_AUDIO_DIR / "output.wav"
        self._audio_buffer = []

    def record_input(self) -> bool:
        """Invokes recorder module to capture voice audio to recordings/input.wav.
        
        Returns:
            True if recording succeeded and captured speech, False otherwise.
        """
        # Timing recording
        start_time = time.time()
        logger.info("AudioService: Starting voice capture process...")
        
        success = record_audio(self.input_wav)
        
        elapsed = time.time() - start_time
        logger.info(f"AudioService: Recording session ended. Elapsed recorder time: {elapsed:.2f} seconds.")
        return success

    def record_input_fixed(self, duration: float = 5.0) -> bool:
        """Records microphone sound for a fixed duration of seconds.
        
        Returns:
            True if recording succeeded, False otherwise.
        """
        start_time = time.time()
        logger.info(f"AudioService: Starting fixed recording (duration {duration}s)...")
        
        success = record_fixed_duration(self.input_wav, duration=duration)
        
        elapsed = time.time() - start_time
        logger.info(f"AudioService: Fixed recording ended. Elapsed time: {elapsed:.2f} seconds.")
        return success

    def record_input_vad(self, silence_seconds: float = 1.5, timeout: float = 15.0) -> bool:
        """Records audio from microphone, automatically ending on silence (VAD).
        
        Returns:
            True if recording succeeded with user speech, False otherwise.
        """
        start_time = time.time()
        logger.info(f"AudioService: Starting VAD voice capture (silence_seconds={silence_seconds})...")
        
        success = record_vad(self.input_wav, silence_seconds=silence_seconds, timeout=timeout)
        
        elapsed = time.time() - start_time
        logger.info(f"AudioService: VAD voice capture finished. Elapsed time: {elapsed:.2f} seconds.")
        return success



    def transcribe_input(self) -> tuple[str, float]:
        """Transcribes the captured input.wav using Faster-Whisper.
        
        Returns:
            Tuple of (transcribed_text_string, transcription_time_seconds).
        """
        if not self.input_wav.exists():
            logger.warning("AudioService: No input.wav file to transcribe.")
            return "", 0.0
            
        start_time = time.time()
        logger.info("AudioService: Initializing transcription...")
        
        text = transcribe_audio(self.input_wav)
        
        elapsed = time.time() - start_time
        logger.info(f"AudioService: Whisper transcription finished in {elapsed:.2f} seconds.")
        return text, elapsed

    def play_assistant_speech(self, text: str) -> tuple[bool, float]:
        """Runs Piper TTS on text, saves output, and plays it back through the speakers.
        
        Args:
            text: Response message from LLM to speak.
            
        Returns:
            Tuple of (success_status, tts_generation_time_seconds).
        """
        start_time = time.time()
        logger.info("AudioService: Generating assistant speech file via Piper...")
        
        # 1. Generate text to speech WAV file
        tts_success = text_to_speech(text, self.output_wav)
        tts_time = time.time() - start_time
        
        if not tts_success:
            logger.error("AudioService: Piper speech generation failed.")
            return False, tts_time
            
        logger.info(f"AudioService: Piper generation finished in {tts_time:.2f} seconds. Starting speaker playback...")
        
        # 2. Play the generated WAV file through speakers
        playback_success = play_audio(self.output_wav)
        
        return playback_success, tts_time
