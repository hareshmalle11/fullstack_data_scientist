# System-wide constants

DB_MESSAGES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL
);
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful offline voice assistant.\n"
    "Speak naturally.\n"
    "Keep answers concise.\n"
    "Avoid code unless explicitly requested.\n"
    "Remember context."
)

# Audio format defaults for wav recording
WAV_SUBTYPE = "PCM_16"

# ANSI Colors for premium CLI styling
class CLIColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
