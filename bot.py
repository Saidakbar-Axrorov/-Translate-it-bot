import asyncio
import logging
import os
from typing import Optional

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

# Environment variables (no hardcoded secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("translator-bot")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text from English "
    "to Uzbek. If the input is not English, do not return anything. "
    "Preserve formatting, emojis, and line breaks. Do not add explanations. "
    "Output only the translation."
    "Output should be in uzbek alphabet, not in russian alphabet"
)


def translate_sync(text: str, safety_identifier: Optional[str] = None) -> str:
    if not text.strip():
        return ""

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        ],
        safety_identifier=safety_identifier,
    )

    return response.output_text.strip()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    user_id = None
    if update.effective_user:
        user_id = str(update.effective_user.id)

    try:
        loop = asyncio.get_running_loop()
        translation = await loop.run_in_executor(
            None, translate_sync, message.text, user_id
        )
        if translation:
            await message.reply_text(translation)
    except Exception:
        logger.exception("Translation failed")
        await message.reply_text("Sorry, translation failed. Please try again.")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if "openrouter.ai" in OPENAI_BASE_URL and not OPENAI_API_KEY.startswith("sk-or-v1"):
        raise RuntimeError(
            "OPENAI_BASE_URL points to OpenRouter, but OPENAI_API_KEY does not look like an OpenRouter key (sk-or-v1...)."
        )
    if "api.openai.com" in OPENAI_BASE_URL and OPENAI_API_KEY.startswith("sk-or-v1"):
        raise RuntimeError(
            "OPENAI_BASE_URL points to OpenAI, but OPENAI_API_KEY looks like an OpenRouter key (sk-or-v1...)."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # For private chats and groups
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # For channel posts
    app.add_handler(
        MessageHandler(
            filters.UpdateType.CHANNEL_POST & filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    logger.info("Translator bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
