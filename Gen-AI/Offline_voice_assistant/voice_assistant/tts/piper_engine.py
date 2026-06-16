import subprocess
import os
from pathlib import Path
from voice_assistant.config.settings import Settings
from voice_assistant.utils.logger import logger
from voice_assistant.utils.exceptions import PiperMissingException

def text_to_speech(text: str, output_wav_path: Path) -> bool:
    """Converts a text string to a WAV audio file using the local Piper executable.
    
    Args:
        text: Text response to speak.
        output_wav_path: Target WAV file location to generate.
        
    Returns:
        True if TTS succeeds, False otherwise.
    """
    piper_exe = Settings.PIPER_EXE_PATH
    voice_model = Settings.PIPER_VOICE_PATH
    
    # Check if files exist
    if not piper_exe.exists():
        logger.error(f"Piper TTS executable not found at: {piper_exe}")
        raise PiperMissingException(f"Piper executable is missing at: {piper_exe}. Check configuration and path.")
        
    if not voice_model.exists():
        logger.error(f"Piper voice ONNX model not found at: {voice_model}")
        raise PiperMissingException(f"Piper voice model file is missing at: {voice_model}.")
        
    # Ensure generated_audio folder exists
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Synthesizing speech via Piper. Target file: {output_wav_path}")
    logger.info(f"Input text (length {len(text)}): '{text}'")
    
    try:
        # We run the command with piper.exe directory as cwd to ensure it loads local dlls.
        # Format command: piper.exe --model model.onnx --output_file output.wav
        cmd = [
            str(piper_exe),
            "--model", str(voice_model),
            "--output_file", str(output_wav_path)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(piper_exe.parent),
            text=True,
            encoding="utf-8"
        )
        
        # Send text through stdin and wait for execution to finish
        stdout_data, stderr_data = process.communicate(input=text)
        
        if process.returncode != 0:
            logger.error(f"Piper process exited with code {process.returncode}.")
            logger.error(f"Piper stderr: {stderr_data}")
            return False
            
        logger.info("Piper speech synthesis completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Exception during Piper speech synthesis execution: {e}")
        return False
