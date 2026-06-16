class VoiceAssistantException(Exception):
    """Base exception for all voice assistant errors."""
    pass

class MicrophoneMissingException(VoiceAssistantException):
    """Raised when no active input microphone device can be found."""
    def __init__(self, message="Microphone device not found or inactive."):
        super().__init__(message)

class WhisperFailureException(VoiceAssistantException):
    """Raised when Faster-Whisper model loading or transcription fails."""
    def __init__(self, message="Whisper transcription engine failed."):
        super().__init__(message)

class OllamaNotRunningException(VoiceAssistantException):
    """Raised when Ollama local daemon is not running or accessible."""
    def __init__(self, message="Ollama server is not running on http://localhost:11434."):
        super().__init__(message)

class DatabaseLockedException(VoiceAssistantException):
    """Raised when the SQLite database file cannot be accessed or written to."""
    def __init__(self, message="Database is locked or cannot be accessed."):
        super().__init__(message)

class PiperMissingException(VoiceAssistantException):
    """Raised when the Piper executable or voice ONNX files are missing."""
    def __init__(self, message="Piper TTS executable or voice model file not found."):
        super().__init__(message)
