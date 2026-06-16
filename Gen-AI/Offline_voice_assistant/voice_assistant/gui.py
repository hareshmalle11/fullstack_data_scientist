import sys
import time
from pathlib import Path
import streamlit as st

# Add the parent directory of 'voice_assistant' to sys.path to prevent import errors on different run directories.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure stdout and stderr to UTF-8 on Windows to prevent terminal formatting errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from voice_assistant.config.settings import Settings
from voice_assistant.llm.ollama_client import check_connection, check_model_exists
from voice_assistant.memory.database import create_database, get_connection
from voice_assistant.services.memory_service import MemoryService
from voice_assistant.services.chat_service import ChatService
from voice_assistant.services.audio_service import AudioService
from voice_assistant.tts.piper_engine import text_to_speech
from voice_assistant.tts.speaker import play_audio, stop_audio
from voice_assistant.utils.exceptions import MicrophoneMissingException
import sounddevice as sd
import soundfile as sf
import numpy as np

# Page Configuration for Premium UI styling
st.set_page_config(
    page_title="Offline AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme customizations & custom CSS for Animated Voice Waves
st.markdown("""
<style>
    .reportview-container {
        background: #111216;
    }
    .wave-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 120px;
        gap: 10px;
        margin: 30px 0;
        background-color: #1a1c23;
        border-radius: 12px;
        padding: 20px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .wave-bar {
        width: 8px;
        height: 20px;
        border-radius: 4px;
        animation: bounce 0.6s ease-in-out infinite alternate;
    }
    .blue {
        background: linear-gradient(135deg, #1e88e5, #42a5f5);
        box-shadow: 0 0 15px rgba(33, 150, 243, 0.6);
    }
    .orange {
        background: linear-gradient(135deg, #fb8c00, #ffb74d);
        box-shadow: 0 0 15px rgba(251, 140, 0, 0.6);
    }
    .purple {
        background: linear-gradient(135deg, #8e24aa, #ba68c8);
        box-shadow: 0 0 15px rgba(142, 36, 170, 0.6);
    }
    @keyframes bounce {
        from { height: 15px; transform: scaleY(1); }
        to { height: 90px; transform: scaleY(1.4); }
    }
    .status-text {
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 10px;
    }
    .voice-bubble-container {
        background-color: #1e2029;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Fetch unique session IDs from database
def get_all_sessions() -> list:
    conn = None
    sessions = []
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM messages ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        sessions = [row["session_id"] for row in rows]
    except Exception:
        pass
    finally:
        if conn:
            conn.close()
    return sessions

# 1. Initialize DB & Session States
create_database()

if "initialized" not in st.session_state:
    st.session_state.memory_service = MemoryService()
    st.session_state.chat_service = ChatService()
    st.session_state.audio_service = AudioService()
    st.session_state.initialized = True
    st.session_state.autoplay_src = None
    st.session_state.session_version = 0
    st.session_state.recording_active = False
    st.session_state.recording_stream = None
    
    # Voice-Only Mode states
    st.session_state.convo_state = "idle" # idle, listening, thinking, speaking
    st.session_state.last_user_text = ""
    st.session_state.last_reply_text = ""

mem_service = st.session_state.memory_service
chat_service = st.session_state.chat_service
audio_service = st.session_state.audio_service

# Manual Record helpers using st.session_state.audio_service._audio_buffer
def start_manual_recording():
    audio_service._audio_buffer = []
    
    def callback(indata, frames, time_info, status):
        # Appending to the buffer inside st.session_state.audio_service is thread-safe
        audio_service._audio_buffer.append(indata.copy())
        
    try:
        stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype='int16',
            callback=callback
        )
        stream.start()
        st.session_state.recording_stream = stream
        st.session_state.recording_active = True
    except Exception as e:
        st.error(f"Failed to start recording: {e}")
        st.session_state.recording_active = False

def stop_manual_recording() -> bool:
    stream = st.session_state.recording_stream
    st.session_state.recording_active = False
    st.session_state.recording_stream = None
    
    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            st.error(f"Error stopping audio input stream: {e}")
            return False
            
    if not audio_service._audio_buffer:
        return False
        
    raw_audio = np.concatenate(audio_service._audio_buffer, axis=0)
    # Clear buffer immediately after extraction
    audio_service._audio_buffer = []
    
    from voice_assistant.recorder.audio_utils import normalize_audio, is_silent
    
    if is_silent(raw_audio):
        st.warning("Silence detected. Recording discarded.")
        return False
        
    normalized = normalize_audio(raw_audio)
    audio_service.input_wav.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import scipy.io.wavfile as wav
        wav.write(str(audio_service.input_wav), 16000, normalized)
        return True
    except Exception as e:
        st.error(f"Failed to write audio WAV file: {e}")
        return False

# Sidebar Navigation & Settings Panel
with st.sidebar:
    st.markdown(f"## 🎙️ Control Panel")
    
    # 2. Startup Diagnostics Check in Sidebar
    st.markdown("### 🔍 System Diagnostics")
    
    # Run status checks
    ollama_ok = check_connection()
    model_ok = check_model_exists(Settings.MODEL_NAME) if ollama_ok else False
    
    try:
        mic_ok = len(sd.query_devices(kind='input')) > 0
    except Exception:
        mic_ok = False
        
    piper_ok = Settings.PIPER_EXE_PATH.exists() and Settings.PIPER_VOICE_PATH.exists()
    db_ok = Settings.DB_PATH.exists()
    
    def render_diagnostic_row(label: str, status: bool):
        badge = "🟢" if status else "🔴"
        text = "Ready" if status else "Error"
        st.markdown(f"{badge} **{label}**: {text}")

    render_diagnostic_row("Ollama Server", ollama_ok)
    render_diagnostic_row("Qwen2.5 Model", model_ok)
    render_diagnostic_row("Microphone Device", mic_ok)
    render_diagnostic_row("Piper TTS Binary", piper_ok)
    render_diagnostic_row("SQLite Database", db_ok)
    
    st.divider()
    
    # 3. Session Management UI
    st.markdown("### 💬 Conversation Session")
    
    # Start new session
    if st.button("✨ Start New Session", use_container_width=True):
        new_sid = mem_service.start_new_session()
        st.session_version += 1
        st.session_state.autoplay_src = None
        st.session_state.convo_state = "idle"
        st.session_state.last_user_text = ""
        st.session_state.last_reply_text = ""
        st.rerun()
        
    # Dropdown to load history
    all_sessions = get_all_sessions()
    if mem_service.session_id not in all_sessions:
        all_sessions.insert(0, mem_service.session_id)
        
    try:
        current_index = all_sessions.index(mem_service.session_id)
    except ValueError:
        current_index = 0
        
    selected_session = st.selectbox(
        "Select Past Chat:",
        options=all_sessions,
        index=current_index,
        key=f"session_selector_{st.session_state.session_version}"
    )
    
    # If dropdown session changes, switch session ID
    if selected_session != mem_service.session_id:
        st.session_state.memory_service = MemoryService(session_id=selected_session)
        st.session_state.autoplay_src = None
        st.session_state.convo_state = "idle"
        st.session_state.last_user_text = ""
        st.session_state.last_reply_text = ""
        st.rerun()
        
    st.divider()
    st.markdown("ℹ️ *This assistant runs completely offline on your local system.*")

# Main Page Design: Tabs split
tab_voice, tab_chat = st.tabs(["🎙️ Voice-Only Live Mode", "💬 Text & Audio Chat Mode"])

# ==================== TAB 1: VOICE-ONLY LIVE MODE ====================
with tab_voice:
    st.markdown("<h2 style='text-align: center;'>🎙️ Voice Conversation Agent</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Hold <b>Spacebar</b> to speak, and release it when you are done.<br>(Or use the manual buttons below)</p>", unsafe_allow_html=True)
    
    # Inject Push-to-Talk JS (Spacebar holds to speak, releases to stop/process)
    import streamlit.components.v1 as components
    components.html("""
    <script>
    const doc = window.parent.document;
    if (!window.parent.__spacebar_handlers_initialized__) {
        window.parent.__spacebar_handlers_initialized__ = true;
        
        window.parent.document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                const active = doc.activeElement;
                const isTyping = active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable;
                if (!isTyping) {
                    if (e.repeat) return;
                    e.preventDefault();
                    
                    // First try to find and click "Interrupt / Speak Now" if the assistant is speaking
                    const interruptBtn = Array.from(doc.querySelectorAll('button')).find(btn => 
                        btn.textContent.includes('Interrupt')
                    );
                    if (interruptBtn) {
                        interruptBtn.click();
                    } else {
                        // Otherwise try to find and click "Start Speaking"
                        const startBtn = Array.from(doc.querySelectorAll('button')).find(btn => 
                            btn.textContent.includes('Start Speaking')
                        );
                        if (startBtn) {
                            startBtn.click();
                        }
                    }
                }
            }
        });
        
        window.parent.document.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                const active = doc.activeElement;
                const isTyping = active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable;
                if (!isTyping) {
                    // Try to find and click "Stop & Process"
                    const stopBtn = Array.from(doc.querySelectorAll('button')).find(btn => 
                        btn.textContent.includes('Stop & Process')
                    );
                    if (stopBtn) {
                        stopBtn.click();
                    }
                }
            }
        });
    }
    </script>
    """, height=0)
    
    st.divider()

    # Main Status Display & Visual Wave Animations
    status_box = st.empty()
    wave_box = st.empty()
    control_box = st.empty()
    
    # Display conversation text container (the proof that it hears and answers)
    if st.session_state.last_user_text or st.session_state.last_reply_text:
        st.markdown("<div class='voice-bubble-container'>", unsafe_allow_html=True)
        if st.session_state.last_user_text:
            st.markdown(f"🗣️ **You said:** *\"{st.session_state.last_user_text}\"*")
        if st.session_state.last_reply_text:
            st.markdown(f"🤖 **Assistant:** {st.session_state.last_reply_text}")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.convo_state == "idle":
        status_box.markdown("<p class='status-text' style='color: #888;'>Ready. Click button below to start.</p>", unsafe_allow_html=True)
        
        if control_box.button("🎙️ Start Speaking", use_container_width=True, type="primary", key="voice_start_speaking_btn"):
            if not mic_ok:
                st.error("Cannot start: Microphone is offline.")
            else:
                start_manual_recording()
                st.session_state.convo_state = "listening"
                st.rerun()
                
    elif st.session_state.convo_state == "listening":
        status_box.markdown("<p class='status-text' style='color: #42a5f5;'>🎙️ Recording... Speak now.</p>", unsafe_allow_html=True)
        
        # Show animated blue wave
        wave_box.markdown("""
        <div class='wave-container'>
            <div class='wave-bar blue' style='animation-delay: 0.1s;'></div>
            <div class='wave-bar blue' style='animation-delay: 0.2s;'></div>
            <div class='wave-bar blue' style='animation-delay: 0.3s;'></div>
            <div class='wave-bar blue' style='animation-delay: 0.4s;'></div>
            <div class='wave-bar blue' style='animation-delay: 0.5s;'></div>
        </div>
        """, unsafe_allow_html=True)
        
        if control_box.button("⏹️ Stop & Process", use_container_width=True, type="primary", key="voice_stop_speaking_btn"):
            if stop_manual_recording():
                st.session_state.convo_state = "thinking"
            else:
                st.session_state.convo_state = "idle"
            st.rerun()
            
    elif st.session_state.convo_state == "thinking":
        status_box.markdown("<p class='status-text' style='color: #ba68c8;'>⚡ Thinking...</p>", unsafe_allow_html=True)
        
        # Show animated purple wave
        wave_box.markdown("""
        <div class='wave-container'>
            <div class='wave-bar purple' style='animation-delay: 0.1s;'></div>
            <div class='wave-bar purple' style='animation-delay: 0.2s;'></div>
            <div class='wave-bar purple' style='animation-delay: 0.3s;'></div>
            <div class='wave-bar purple' style='animation-delay: 0.4s;'></div>
            <div class='wave-bar purple' style='animation-delay: 0.5s;'></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Transcribe Whisper WAV
        user_text, stt_time = audio_service.transcribe_input()
        
        if user_text:
            st.session_state.last_user_text = user_text
            st.toast(f"STT: '{user_text}'", icon="🗣️")
            
            # 2. Get LLM response
            history = mem_service.get_history()
            reply_text, llm_time = chat_service.get_assistant_reply(user_text, history)
            
            st.session_state.last_reply_text = reply_text
            
            # 3. Save turns to SQLite + RAM Cache
            mem_service.add_message("user", user_text)
            mem_service.add_message("assistant", reply_text)
            
            # 4. Generate speech synthesis WAV file
            text_to_speech(reply_text, audio_service.output_wav)
            
            st.session_state.convo_state = "speaking"
            st.rerun()
        else:
            st.warning("No speech could be transcribed.")
            st.session_state.convo_state = "idle"
            st.rerun()
            
    elif st.session_state.convo_state == "speaking":
        status_box.markdown("<p class='status-text' style='color: #fb8c00;'>🔊 Speaking response...</p>", unsafe_allow_html=True)
        
        # Show animated orange wave
        wave_box.markdown("""
        <div class='wave-container'>
            <div class='wave-bar orange' style='animation-delay: 0.1s;'></div>
            <div class='wave-bar orange' style='animation-delay: 0.2s;'></div>
            <div class='wave-bar orange' style='animation-delay: 0.3s;'></div>
            <div class='wave-bar orange' style='animation-delay: 0.4s;'></div>
            <div class='wave-bar orange' style='animation-delay: 0.5s;'></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Non-blocking speaker playback
        play_audio(audio_service.output_wav, blocking=False)
        
        # Show interrupt button
        if control_box.button("⏹️ Interrupt / Speak Now", use_container_width=True, type="primary", key="voice_interrupt_btn"):
            stop_audio()
            st.session_state.convo_state = "idle"
            st.rerun()
            
        # Read soundfile length to wait
        try:
            data, fs = sf.read(str(audio_service.output_wav))
            duration = len(data) / fs
        except Exception:
            duration = 3.0
            
        # Sleep in intervals waiting for sound to finish
        start_wait = time.time()
        while time.time() - start_wait < duration:
            time.sleep(0.1)
            
        st.session_state.convo_state = "idle"
        st.rerun()

# ==================== TAB 2: TEXT & AUDIO CHAT MODE ====================
with tab_chat:
    st.markdown("### 💬 Chat Sandbox")
    
    # Display message bubbles
    for msg in mem_service.get_history():
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)
            
    # Playback helper widget
    if st.session_state.autoplay_src:
        st.audio(st.session_state.autoplay_src, format="audio/wav", autoplay=True)
        st.session_state.autoplay_src = None
        
    user_text = None
    
    col_input, col_voice = st.columns([8, 2], gap="small")
    
    with col_voice:
        # Manual Record button
        if not mic_ok:
            st.button("🎙️ Recorder Offline", disabled=True, use_container_width=True)
        elif st.session_state.recording_active:
            if st.button("⏹️ Stop Manual Record", type="primary", use_container_width=True, key="chat_stop_record_btn"):
                if stop_manual_recording():
                    with st.spinner("⚡ Transcribing..."):
                        transcript, stt_time = audio_service.transcribe_input()
                    if transcript:
                        user_text = transcript
                        st.toast(f"Transcribed: '{transcript}'", icon="🗣️")
                st.rerun()
        else:
            if st.button("🎙️ Start Manual Record", use_container_width=True, key="chat_start_record_btn"):
                start_manual_recording()
                st.rerun()
                
    with col_input:
        typed_input = st.chat_input("Type your message here...", key="chat_tab_input")
        if typed_input:
            user_text = typed_input
            
    if user_text:
        # User message
        with st.chat_message("user"):
            st.markdown(user_text)
        mem_service.add_message("user", user_text)
        
        # Assistant message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🤖 Thinking..."):
                history = mem_service.get_history()[:-1]
                reply_text, llm_time = chat_service.get_assistant_reply(user_text, history)
                
            message_placeholder.markdown(reply_text)
            mem_service.add_message("assistant", reply_text)
            
            with st.spinner("⚡ Synthesizing response..."):
                tts_success = text_to_speech(reply_text, audio_service.output_wav)
                
            if tts_success:
                st.session_state.autoplay_src = str(audio_service.output_wav)
                st.rerun()
            else:
                st.error("Speech synthesis failed.")
