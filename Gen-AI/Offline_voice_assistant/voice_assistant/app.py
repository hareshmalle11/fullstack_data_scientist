import sys
import time
from pathlib import Path

# Add the parent directory of 'voice_assistant' to sys.path to prevent import errors on different run directories.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure stdout and stderr to use UTF-8 on Windows to prevent encoding crashes when printing emojis.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from voice_assistant.config.settings import Settings

from voice_assistant.config.constants import CLIColors
from voice_assistant.utils.startup_checks import run_startup_checks
from voice_assistant.memory.database import create_database
from voice_assistant.services.memory_service import MemoryService
from voice_assistant.services.chat_service import ChatService
from voice_assistant.services.audio_service import AudioService
from voice_assistant.utils.logger import logger


def display_banner():
    """Prints a premium ANSI styled banner for the assistant."""
    banner = f"""
{CLIColors.BLUE}======================================================================
{CLIColors.BOLD}{CLIColors.CYAN}       🎙️  OFFLINE AI VOICE ASSISTANT (Qwen2.5-Coder:3B)  🎙️
{CLIColors.ENDC}{CLIColors.BLUE}======================================================================
  • Fully Local & Private (No Cloud, No External APIs)
  • Speech-To-Text: Faster-Whisper
  • LLM Engine: Ollama (qwen2.5-coder:3b)
  • Persistent Memory: SQLite + RAM Cache
  • Text-To-Speech: Piper TTS
======================================================================{CLIColors.ENDC}
"""
    print(banner)

def main():
    # 1. Run Startup Diagnostics
    try:
        run_startup_checks()
    except Exception as e:
        print(f"\n{CLIColors.FAIL}{CLIColors.BOLD}❌ Startup Check Failure:{CLIColors.ENDC} {e}")
        logger.critical(f"System startup checks failed: {e}")
        print(f"{CLIColors.YELLOW}Please resolve the errors above and run again.{CLIColors.ENDC}")
        sys.exit(1)

    # 2. Initialize database
    try:
        create_database()
    except Exception as e:
        print(f"❌ Failed to initialize SQLite database: {e}")
        sys.exit(1)

    # 3. Load services
    logger.info("Initializing services...")
    memory_service = MemoryService()
    chat_service = ChatService()
    audio_service = AudioService()
    logger.info("All services initialized.")

    display_banner()
    print(f"{CLIColors.GREEN}Active Session: {CLIColors.BOLD}{memory_service.session_id}{CLIColors.ENDC}")
    print(f"Type {CLIColors.BOLD}'new'{CLIColors.ENDC} to clear history, {CLIColors.BOLD}'exit'{CLIColors.ENDC} to quit, or press {CLIColors.BOLD}Enter{CLIColors.ENDC} to speak.")

    while True:
        try:
            print(f"\n{CLIColors.BLUE}--- Ready ---{CLIColors.ENDC}")
            command = input(f"Press Enter to record voice, or type command: ").strip().lower()

            if command == "exit":
                print(f"\n{CLIColors.CYAN}Goodbye!{CLIColors.ENDC}")
                logger.info("User requested exit. Exiting assistant.")
                break
            elif command == "new":
                new_sid = memory_service.start_new_session()
                print(f"✨ Started new session: {CLIColors.BOLD}{new_sid}{CLIColors.ENDC}")
                continue
            elif command != "":
                # User typed a text message instead of recording
                user_text = command
                logger.info(f"User typed message: '{user_text}'")
                stt_time = 0.0
            else:
                # 4. Record audio from mic
                record_ok = audio_service.record_input()
                if not record_ok:
                    continue
                
                # 5. Transcribe audio to text
                print(f"{CLIColors.YELLOW}⚡ Transcribing speech...{CLIColors.ENDC}")
                user_text, stt_time = audio_service.transcribe_input()
                
                if not user_text:
                    print("⚠️ Could not transcribe any speech. Please try again.")
                    continue
                    
                print(f"\n{CLIColors.BOLD}🗣️ You (STT):{CLIColors.ENDC} {user_text} {CLIColors.BLUE}({stt_time:.2f}s){CLIColors.ENDC}")

            # 6. Retrieve recent chat memory context
            history = memory_service.get_history()

            # 7. Get response from LLM
            print(f"{CLIColors.YELLOW}⚡ Thinking...{CLIColors.ENDC}")
            assistant_reply, llm_time = chat_service.get_assistant_reply(user_text, history)
            
            print(f"\n{CLIColors.BOLD}{CLIColors.CYAN}🤖 Assistant:{CLIColors.ENDC} {assistant_reply} {CLIColors.BLUE}({llm_time:.2f}s){CLIColors.ENDC}")

            # 8. Save user and assistant message
            memory_service.add_message("user", user_text)
            memory_service.add_message("assistant", assistant_reply)

            # 9. Convert to speech and play aloud
            print(f"{CLIColors.YELLOW}⚡ Generating voice response...{CLIColors.ENDC}")
            tts_ok, tts_time = audio_service.play_assistant_speech(assistant_reply)
            
            if tts_ok:
                print(f"{CLIColors.GREEN}🔊 Playing speech...{CLIColors.ENDC} {CLIColors.BLUE}(TTS Gen: {tts_time:.2f}s){CLIColors.ENDC}")
            else:
                print(f"{CLIColors.FAIL}⚠️ Speech generation or playback failed.{CLIColors.ENDC}")

            # Log consolidated metrics summary
            logger.info(
                f"Metrics | Session={memory_service.session_id} | "
                f"STT={stt_time:.2f}s | LLM={llm_time:.2f}s | TTS={tts_time:.2f}s"
            )

        except KeyboardInterrupt:
            print(f"\n\n{CLIColors.CYAN}Goodbye! (Session interrupted){CLIColors.ENDC}")
            logger.info("Keyboard interrupt caught. Exiting.")
            break
        except Exception as e:
            logger.error(f"Error in main application loop: {e}", exc_info=True)
            print(f"\n{CLIColors.FAIL}❌ An unexpected error occurred: {e}{CLIColors.ENDC}")
            time.sleep(2)

if __name__ == "__main__":
    main()
