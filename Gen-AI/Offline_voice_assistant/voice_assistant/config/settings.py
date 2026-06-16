import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project (voice_assistant/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Set HF HuggingFace configuration environment variables
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables from .env file
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    USE_CUDA = os.getenv("USE_CUDA", "True").lower() in ("true", "1", "yes")
    
    # Resolve relative database path
    raw_db_path = os.getenv("DB_PATH", "database/chat_history.db")
    DB_PATH = BASE_DIR / raw_db_path
    
    # Resolve relative piper paths
    raw_piper_exe = os.getenv("PIPER_EXE_PATH", "tts/piper/piper.exe")
    PIPER_EXE_PATH = BASE_DIR / raw_piper_exe
    
    raw_piper_voice = os.getenv("PIPER_VOICE_PATH", "tts/piper/en_US-lessac-medium.onnx")
    PIPER_VOICE_PATH = BASE_DIR / raw_piper_voice
    
    # Directories
    RECORDINGS_DIR = BASE_DIR / "recordings"
    GENERATED_AUDIO_DIR = BASE_DIR / "generated_audio"
    LOGS_DIR = BASE_DIR / "logs"
    DATABASE_DIR = DB_PATH.parent
    SESSIONS_DIR = BASE_DIR / "sessions"
    PROMPTS_DIR = BASE_DIR / "prompts"
    
    # Audio config defaults
    SAMPLE_RATE = 16000  # 16kHz is standard for Whisper and Piper
    CHANNELS = 1         # Mono
    
    # Log files
    LOG_FILE = LOGS_DIR / "assistant.log"
    
    # Prompts
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
    
    # Session ID tracker
    CURRENT_SESSION_FILE = SESSIONS_DIR / "current_session.txt"

# Ensure crucial directories exist
for directory in [
    Settings.RECORDINGS_DIR,
    Settings.GENERATED_AUDIO_DIR,
    Settings.LOGS_DIR,
    Settings.DATABASE_DIR,
    Settings.SESSIONS_DIR,
    Settings.PROMPTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
