"""
Voice Telegram Bot
===================
A modular Telegram bot that transcribes incoming voice messages, summarizes them with an
LLM (local or cloud), and stores both the raw transcription and the summary in a database.

Architecture Overview
---------------------
• ``Database`` – Thin wrapper around SQLite **(switchable to MySQL later)**
• ``LLMHandler`` – Talks to either a **local** Ollama instance *or* a cloud LLM (e.g. OpenAI)
• ``SpeechToText`` – Converts OGG ➜ WAV (via FFmpeg) and transcribes with **Vosk**
• ``TelegramBot`` – Handles the Telegram Bot API (using ``python‑telegram‑bot`` library)
• ``Utils`` – Misc. helper utilities (static methods only)

Environment variables are read from a ``.env`` file – **change behaviour by editing the file, not the code**.

Example .env
-------------
```
# ─── Telegram ────────────────────────────────────────────────────────────────────
TG_BOT_TOKEN="123456:ABC‑DEF…”

# ─── Database ────────────────────────────────────────────────────────────────────
DB_PATH="bot.db"              # Path to SQLite file (ignored if you later switch to MySQL)

# ─── LLM Settings ────────────────────────────────────────────────────────────────
USE_LOCAL_LLM="True"          # "True" -> Ollama, "False" -> use cloud LLM
OLLAMA_BASE_URL="http://127.0.0.1:11434"  # Where your Ollama server sits
OLLAMA_MODEL="mistral"

OPENAI_API_KEY="sk‑…"         # Only required when USE_LOCAL_LLM=False
OPENAI_MODEL="gpt‑3.5‑turbo"  # or gpt‑4o, etc.

# ─── Speech‑to‑Text ──────────────────────────────────────────────────────────────
VOSK_MODEL_PATH="models/vosk‑small‑ru‑0.22"
FFMPEG_BINARY="ffmpeg"         # Path or command alias

# ─── Misc ────────────────────────────────────────────────────────────────────────
LOG_FILE="bot.log"             # Where to write debug logs

```

Install dependencies
--------------------
```bash
python -m venv .venv && source .venv/bin/activate
pip install python‑telegram‑bot==20.8 python‑dotenv requests vosk ffmpeg‑python openai
```
(Use ``pip install mysql‑connector‑python`` later if/when you migrate the DB.)

Run the bot
-----------
```bash
python voice_telegram_bot.py
```

-------------------------------------------------------------------------------
Below is the **fully‑commented** implementation. Feel free to slice it into
multiple files once you're comfortable – the class boundaries are already set up
for easy extraction.
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import sqlite3
import logging
import datetime as _dt
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from telegram import Update, Voice, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

try:
    import openai  # optional – only needed when USE_LOCAL_LLM == False
except ImportError:
    openai = None  # keep linting happy – runtime guard is in LLMHandler

# ────────────────────────────────────────────────────────────────────────────────
#  Utils
# ────────────────────────────────────────────────────────────────────────────────
class Utils:
    """Collection of helper static methods."""

    @staticmethod
    def now_utc() -> str:
        """Return current UTC time in ISO‑8601 format (without microseconds)."""
        return _dt.datetime.utcnow().replace(microsecond=0).isoformat()

    @staticmethod
    def ensure_parent(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def as_bool(value: str | bool | None) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def configure_logging(log_path: str):
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.FileHandler(log_path, encoding="utf‑8"),
                logging.StreamHandler(),
            ],
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)

# ────────────────────────────────────────────────────────────────────────────────
#  Database Layer
# ────────────────────────────────────────────────────────────────────────────────
class Database:
    """Lightweight wrapper around SQLite – swappable later for MySQL.

    Only *this* class knows specific SQL dialect details. The rest of the code
    interacts through the typed public interface below. To migrate, rewrite
    these methods and keep signatures intact.
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS voice_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_file_id TEXT UNIQUE NOT NULL,
        text        TEXT,
        summarized  TEXT,
        sent_at     TEXT,
        user_id     INTEGER,
        username    TEXT
    );"""

    def __init__(self, path: str):
        Utils.ensure_parent(Path(path))
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency
        self.conn.execute(self._CREATE_TABLE_SQL)
        self.conn.commit()
        logging.info("Connected to SQLite DB at %s", path)

    def save_message(
        self,
        telegram_file_id: str,
        text: str | None,
        summary: str | None,
        sent_at: str,
        user_id: int,
        username: str | None,
    ) -> None:
        """Insert or update a transcription row."""
        self.conn.execute(
            """
            INSERT INTO voice_messages
                (telegram_file_id, text, summarized, sent_at, user_id, username)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(telegram_file_id) DO UPDATE SET
                text=excluded.text,
                summarized=excluded.summarized;
            """,
            (telegram_file_id, text, summary, sent_at, user_id, username),
        )
        self.conn.commit()
        logging.debug("Saved message %s", telegram_file_id)

    # Example of an accessor you might need later.
    def fetch_last_n(self, n: int = 10):
        cur = self.conn.execute(
            "SELECT * FROM voice_messages ORDER BY id DESC LIMIT ?", (n,)
        )
        return cur.fetchall()

# ────────────────────────────────────────────────────────────────────────────────
#  LLM Handler
# ────────────────────────────────────────────────────────────────────────────────
class LLMHandler:
    """Facade that hides whether we call a *local* Ollama model or a *cloud* LLM."""

    def __init__(self, *, use_local: bool, cfg: Dict[str, str]):
        self.use_local = use_local
        self.cfg = cfg
        if not use_local and openai is None:
            raise RuntimeError(
                "OpenAI Python package is not installed but USE_LOCAL_LLM=False." )
        if not use_local:
            openai.api_key = cfg["OPENAI_API_KEY"]

    def summarize(self, text: str) -> str:
        """
        Генерирует резюме с учётом ограничений длины.
        Если результат отклоняется от диапазона >20 %, делаем
        повторную попытку — число попыток берётся из SUMMARY_MAX_TRIES.
        """
        deviation = int(self.cfg.get("SUMMARY_DEVIATION_PERCENT", 20))

        n = len(text)
        if n < 500:
            min_len, max_len = 150, 250
        else:
            min_len, max_len = max(150, n // 5), n // 3  # 5:1 и 3:1

        # сколько раз можно пробовать
        tries = int(self.cfg.get("SUMMARY_MAX_TRIES", 2))
        tries = max(1, tries)  # хотя бы одна

        def _generate() -> str:
            prompt = (
                "Сделай краткое резюме объёмом от "
                f"{min_len} до {max_len} символов (без кавычек, только текст) "
                "для следующего голосового сообщения:\n\n"
                f"{text}"
            )
            return (
                self._ollama_call(prompt)
                if self.use_local
                else self._openai_call(prompt)
            ).strip()

        for attempt in range(tries):
            summary = _generate()
            length = len(summary)
            if min_len * (1-(deviation/100)) <= length <= max_len * (1+(deviation/100)):
                # допуск ±20 % — устраивает
                break
            if attempt == tries - 1:
                # последняя попытка, примем как есть
                logging.warning(
                    "Summary length %d вне диапазона (%d‑%d) даже после %d попыток",
                    length, min_len, max_len, tries
                )

        return summary

    # ─── Private helpers ───────────────────────────────────────────────────────
    def _ollama_call(self, prompt: str) -> str:
        url = f"{self.cfg['OLLAMA_BASE_URL']}/api/generate"
        payload = {"model": self.cfg["OLLAMA_MODEL"], "prompt": prompt, "stream": False}
        logging.debug("Ollama request: %s", json.dumps(payload)[:200])
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        logging.info("Ollama summary len=%d", len(summary))
        return summary

    def _openai_call(self, prompt: str) -> str:
        logging.debug("OpenAI prompt len=%d", len(prompt))
        resp = openai.ChatCompletion.create(
            model=self.cfg["OPENAI_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        summary = resp.choices[0].message.content.strip()
        logging.info("OpenAI summary len=%d", len(summary))
        return summary

# ────────────────────────────────────────────────────────────────────────────────
#  Speech‑to‑Text
# ────────────────────────────────────────────────────────────────────────────────
class SpeechToText:
    """Handles audio conversion + transcription via Vosk.

    Changing STT backend? Replace *only* this class. Maintain public method
    signatures and the rest of the program will keep working. See the inline
    guide below for hints on swapping models (e.g. Whisper, DeepSpeech …).
    """
    SAMPLE_RATE = 16_000  # удобная константа

    def __init__(self, model_path: str, ffmpeg_cmd: str = "ffmpeg"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Vosk model not found: {model_path}")
        self.model = Model(model_path)
        self.ffmpeg_cmd = ffmpeg_cmd
        logging.info("Loaded Vosk model from %s", model_path)

    # ---------------------------------------------------------------------
    #  PUBLIC API
    # ---------------------------------------------------------------------
    def transcribe_ogg_bytes(self, ogg_bytes: bytes) -> str:
        """OGG ➜ WAV ➜ TEXT (returns transcription)."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg:
            ogg.write(ogg_bytes)
            ogg_path = Path(ogg.name)
        wav_path = ogg_path.with_suffix(".wav")
        self._ogg_to_wav(ogg_path, wav_path)
        text = self._wav_to_text(wav_path)
        # cleanup temp files
        ogg_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        return text

    # ---------------------------------------------------------------------
    #  INTERNAL HELPERS
    # ---------------------------------------------------------------------
    def _ogg_to_wav(self, ogg_path: Path, wav_path: Path) -> None:
        """Конвертируем Ogg/Opus → WAV 16 kHz mono s16le."""
        cmd = [
            self.ffmpeg_cmd,  # берётся из env‑переменной FFMPEG_BINARY
            "-i", str(ogg_path),
            "-ar", str(self.SAMPLE_RATE),  # частота дискретизации
            "-ac", "1",  # mono
            "-c:a", "pcm_s16le",  # несжатый PCM
            "-y",  # overwrite
            "-loglevel", "quiet",  # тишина в логи FFmpeg
            str(wav_path),
        ]
        subprocess.run(cmd, check=True)

    def _wav_to_text(self, wav_path: Path) -> str:
        """Читает WAV и возвращает финальный текст Vosk."""
        rec = KaldiRecognizer(self.model, self.SAMPLE_RATE)
        with open(wav_path, "rb") as wf:
            while chunk := wf.read(4000):
                rec.AcceptWaveform(chunk)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()

    # ---------------------------------------------------------------------
    #  HOW TO SWITCH TO ANOTHER STT BACKEND (eg. Whisper)  ─────────────────
    # ---------------------------------------------------------------------
    # 1. Install the new library (pip install‑U whisper‑cpp …).
    # 2. Delete/replace code inside _wav_to_text() with call into that lib.
    # 3. Make sure transcribe_ogg_bytes() still returns *plain string*.
    # 4. Keep signatures unchanged ⇒ rest of the program stays happy.

# ────────────────────────────────────────────────────────────────────────────────
#  Telegram Bot Wrapper
# ────────────────────────────────────────────────────────────────────────────────
class TelegramBot:
    """Encapsulates all Telegram‑specific logic. Runs the event loop."""

    def __init__(
        self,
        token: str,
        db: Database,
        stt: SpeechToText,
        llm: LLMHandler,
    ):
        self.db = db
        self.stt = stt
        self.llm = llm
        self.app = ApplicationBuilder().token(token).build()
        # Register handler for *voice* messages only.
        self.app.add_handler(
            MessageHandler(filters.VOICE & ~filters.COMMAND, self.on_voice)
        )
        logging.info("Telegram bot initialized – waiting for voice messages…")

    # ---------------------------------------------------------------------
    async def on_voice(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        assert isinstance(update.effective_message.voice, Voice)
        voice: Voice = update.effective_message.voice
        sent_at = Utils.now_utc()
        user = update.effective_user
        file_id = voice.file_id
        logging.info(
            "Voice received: file_id=%s | from=%s", file_id, user.username or user.id
        )
        tg_file = await ctx.bot.get_file(file_id)
        ogg_bytes = await tg_file.download_as_bytearray()

        # 1️⃣ Transcribe
        text = self.stt.transcribe_ogg_bytes(ogg_bytes)
        logging.info("Transcribed %d chars", len(text))

        # 2️⃣ Summarize
        summary = self.llm.summarize(text)

        # 3️⃣ Persist
        self.db.save_message(
            telegram_file_id=file_id,
            text=text,
            summary=summary,
            sent_at=sent_at,
            user_id=user.id,
            username=user.username,
        )

        # 4️⃣ Respond with summary
        await update.message.reply_text(
            f"📌 Резюме: {summary}", parse_mode=constants.ParseMode.HTML
        )

    # ---------------------------------------------------------------------
    def run(self):
        """Kick‑off polling loop (Ctrl‑C to exit)."""
        self.app.run_polling()

# ────────────────────────────────────────────────────────────────────────────────
#  Entry‑Point
# ────────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(dotenv_path='.env')

    # Configure logging *first* so other classes can log.
    Utils.configure_logging(os.getenv("LOG_FILE", "bot.log"))

    # Instantiate sub‑systems using env‑config.
    db = Database(os.getenv("DB_PATH", "bot.db"))

    stt = SpeechToText(
        model_path=os.getenv("VOSK_MODEL_PATH", "models/vosk‑small‑ru‑0.22"),
        ffmpeg_cmd=os.getenv("FFMPEG_BINARY", "ffmpeg"),
    )

    llm = LLMHandler(
        use_local=Utils.as_bool(os.getenv("USE_LOCAL_LLM", "True")),
        cfg=os.environ,  # pass entire dict; handler reads what it needs
    )

    bot = TelegramBot(
        token=os.environ["TG_BOT_TOKEN"],
        db=db,
        stt=stt,
        llm=llm,
    )

    bot.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        raise
