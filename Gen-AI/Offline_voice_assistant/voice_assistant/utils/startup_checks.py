import sounddevice as sd
from pathlib import Path
from voice_assistant.config.settings import Settings
from voice_assistant.config.constants import CLIColors
from voice_assistant.llm.ollama_client import check_connection, check_model_exists
from voice_assistant.memory.database import get_connection
from voice_assistant.utils.logger import logger
from voice_assistant.utils.exceptions import (
    OllamaNotRunningException,
    MicrophoneMissingException,
    PiperMissingException,
    DatabaseLockedException
)

def print_status(check_name: str, success: bool, details: str = ""):
    """Helper to format and print startup check results to the console."""
    if success:
        print(f" {CLIColors.GREEN}✓{CLIColors.ENDC} {check_name} {CLIColors.BOLD}{CLIColors.GREEN}Found{CLIColors.ENDC} {details}")
    else:
        print(f" {CLIColors.FAIL}✗{CLIColors.ENDC} {check_name} {CLIColors.BOLD}{CLIColors.FAIL}Failed{CLIColors.ENDC} {details}")

def run_startup_checks() -> bool:
    """Executes validation checks on all essential voice assistant components.
    
    Raises appropriate custom exceptions if an item fails validation.
    """
    print(f"\n{CLIColors.HEADER}{CLIColors.BOLD}🔍 Running System Startup Checks...{CLIColors.ENDC}")
    logger.info("Starting startup verification sequence...")
    
    # 1. Check Ollama server running
    try:
        ollama_ok = check_connection()
        if not ollama_ok:
            print_status("Ollama Running", False, "(Make sure Ollama desktop app is started)")
            raise OllamaNotRunningException("Ollama daemon is unresponsive or not running.")
        print_status("Ollama Running", True)
    except Exception as e:
        if not isinstance(e, OllamaNotRunningException):
            logger.error(f"Error during Ollama connection check: {e}")
            raise OllamaNotRunningException(f"Failed to check Ollama service: {e}") from e
        raise e

    # 2. Check model exists
    model_name = Settings.MODEL_NAME
    model_ok = check_model_exists(model_name)
    if not model_ok:
        print_status(f"Model Exists ({model_name})", False, f"(Run 'ollama pull {model_name}' first)")
        raise OllamaNotRunningException(f"The model '{model_name}' was not found in local Ollama.")
    print_status(f"Model Exists ({model_name})", True)

    # 3. Check microphone
    try:
        # Query devices and check for any input device
        input_devices = sd.query_devices(kind='input')
        if not input_devices:
            print_status("Microphone Found", False)
            raise MicrophoneMissingException()
        print_status("Microphone Found", True, f"({input_devices.get('name')})")
    except Exception as e:
        if not isinstance(e, MicrophoneMissingException):
            print_status("Microphone Found", False, f"(Error: {e})")
            raise MicrophoneMissingException(f"Microphone verification error: {e}") from e
        raise e

    # 4. Check Piper executable and voice
    piper_exe = Settings.PIPER_EXE_PATH
    piper_voice = Settings.PIPER_VOICE_PATH
    
    # Check piper executable
    if not piper_exe.exists():
        print_status("Piper Executable Found", False, f"(Missing at {piper_exe})")
        raise PiperMissingException(f"Piper executable not found at: {piper_exe}")
        
    # Check voice file
    if not piper_voice.exists():
        print_status("Piper Voice Model Found", False, f"(Missing at {piper_voice})")
        raise PiperMissingException(f"Piper voice model file not found at: {piper_voice}")
        
    print_status("Piper Engine & Voice", True, f"(Binary: {piper_exe.name}, Voice: {piper_voice.name})")

    # 5. Check Database
    conn = None
    try:
        conn = get_connection()
        print_status("Database Ready", True, f"({Settings.DB_PATH.name})")
    except Exception as e:
        print_status("Database Ready", False, f"(Error: {e})")
        raise DatabaseLockedException(f"Failed to connect or unlock database: {e}") from e
    finally:
        if conn:
            conn.close()

    print(f"{CLIColors.GREEN}{CLIColors.BOLD}🎉 All startup checks passed successfully! System is ready.{CLIColors.ENDC}\n")
    logger.info("All startup checks passed successfully.")
    return True
