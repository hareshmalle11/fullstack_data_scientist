# Local Offline AI Voice Assistant

A fully local, private voice assistant running offline on Windows, powered by **Faster-Whisper**, **Ollama (Qwen2.5-Coder:3B)**, **SQLite**, and **Piper TTS**.

---

## 🛠️ Tech Stack & Key Features

*   **Speech-to-Text (STT)**: `faster-whisper` (utilizes GPU/CUDA acceleration with graceful CPU fallback).
*   **LLM Engine**: Local `Ollama` daemon running the `qwen2.5-coder:3b` model.
*   **Persistent Memory**: Local SQLite database storage + cached RAM conversation buffer.
*   **Text-to-Speech (TTS)**: `Piper TTS` (invoked locally via subprocess).
*   **Audio Capture**: `sounddevice` and `numpy` using keypress controls (Press Enter to start/stop).
*   **Playback**: Native Windows `winsound` playback.
*   **Self-Diagnostics**: Comprehensive startup checks validating audio hardware, Ollama connectivity, model downloads, database lockouts, and Piper path configurations.

---

## 📂 Project Structure

```text
voice_assistant/
├── app.py                     # Main orchestrator & CLI entry point
├── .env                       # Local environment configurations
├── config/
│   ├── settings.py            # Environment configurations parser
│   └── constants.py           # Database schemas, styling colors, and defaults
├── recorder/
│   ├── recorder.py            # Captures audio using enter toggle controls
│   └── audio_utils.py         # Normalization & silence filters
├── stt/
│   └── whisper_engine.py      # Faster-Whisper transcriber (GPU/CPU fallback)
├── llm/
│   ├── ollama_client.py       # Local Ollama client connector
│   ├── prompt_builder.py      # System instructions & chat messages formatter
│   └── context_manager.py     # Message history size controller
├── memory/
│   ├── database.py            # SQLite connection pool and migrations
│   └── history.py             # SQLite reader/writer queries
├── tts/
│   ├── piper_engine.py        # Subprocess caller for the Piper TTS binary
│   └── speaker.py             # Audio playback via winsound
├── services/
│   ├── audio_service.py       # Orchestrates recording, transcription, and TTS
│   ├── chat_service.py        # Orchestrates prompt creation & LLM queries
│   └── memory_service.py      # Manages the active session cache in RAM and DB
├── utils/
│   ├── exceptions.py          # Custom exceptions
│   ├── logger.py              # Centralized stdout & file logging
│   └── startup_checks.py      # Verification scripts run at boot
├── prompts/
│   └── system_prompt.txt      # System template instructions
├── sessions/
│   └── current_session.txt    # Persisted active session ID
├── recordings/                # Temp directory for voice inputs
├── generated_audio/           # Temp directory for speech responses
├── database/                  # Stores SQLite db files
├── logs/
│   └── assistant.log          # Detailed application runtime logs
├── requirements.txt           # Python dependency file
└── README.md                  # System manual
```

---

## 🚀 Setting Up the Environment

### 1. Pre-requisites
Ensure you have the following installed on your Windows machine:
*   **Python 3.12**
*   **Ollama for Windows**: Download from [ollama.com](https://ollama.com). Ensure the server is running and you have pulled the model:
    ```cmd
    ollama pull qwen2.5-coder:3b
    ```
*   *(Optional)* **NVIDIA CUDA Toolkits & Drivers**: For GPU-accelerated transcription (Faster-Whisper will fall back to CPU automatically if not present).

---

### 2. Install Python Dependencies
Open PowerShell or Command Prompt in the `voice_assistant/` directory:
```cmd
pip install -r requirements.txt
```

---

### 3. Set Up Piper TTS (Manual Installation)
Since automatic binary downloads are removed for stability, set up Piper manually inside the project:

1.  Download `piper_windows_amd64.zip` from the [Piper GitHub Releases Page](https://github.com/rhasspy/piper/releases/tag/v1.2.0).
2.  Extract the ZIP contents. Rename the folder to `piper` and place it in the `voice_assistant/tts/` folder.
3.  Ensure `piper.exe` is located at `voice_assistant/tts/piper/piper.exe`.
4.  Download the voice model `.onnx` and configuration `.onnx.json` files from the Hugging Face [rhasspy/piper-voices repository](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium).
    *   [en_US-lessac-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true)
    *   [en_US-lessac-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true)
5.  Place both files inside the `voice_assistant/tts/piper/` folder.
    *   `voice_assistant/tts/piper/en_US-lessac-medium.onnx`
    *   `voice_assistant/tts/piper/en_US-lessac-medium.onnx.json`

Your `voice_assistant/tts/piper/` folder should look like:
```text
voice_assistant/tts/piper/
├── piper.exe
├── en_US-lessac-medium.onnx
├── en_US-lessac-medium.onnx.json
├── ... (various libraries, e.g. onnxruntime.dll)
```

---

### 4. Configuration Check (.env)
The system reads local file parameters using the relative path mapping in `voice_assistant/.env`. Ensure the settings match your file setup:
```env
MODEL_NAME=qwen2.5-coder:3b
WHISPER_MODEL=small
USE_CUDA=True
DB_PATH=database/chat_history.db
PIPER_EXE_PATH=tts/piper/piper.exe
PIPER_VOICE_PATH=tts/piper/en_US-lessac-medium.onnx
```

---

## 🏃 Running the Assistant

Execute the app.py file from the parent folder of `voice_assistant/`:
```cmd
python -m voice_assistant.app
```
*(Or navigate inside the `voice_assistant/` directory and run: `python app.py`)*

### Console Interaction Guide:
1.  On startup, the system runs all diagnostic checks.
2.  Press **Enter** to start voice recording.
3.  Speak into the microphone.
4.  Press **Enter** again to stop recording.
5.  The assistant will transcribe your voice, feed it to Ollama with past conversation history, show the response, generate voice output, and play the response.
6.  Type `new` at the prompt to wipe cache and start a fresh conversation session.
7.  Type `exit` or press `Ctrl + C` to close the assistant.
