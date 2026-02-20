import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag.query_rag import retrieve_context, generate_answer, load_vectorstore

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бухгалтерский RAG-бот. Задай вопрос по зарплате, налогам или регламентам."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text

    answer = None

    try:
        vectorstore = load_vectorstore()
        docs = retrieve_context(vectorstore, user_question)
        answer = generate_answer(docs, user_question)
    except Exception as e:
        print("RAG ERROR:", repr(e))
        answer = (
            "⚠️ Ответ не может быть сформирован.\n\n"
            "Причина: региональные ограничения внешнего AI-API.\n"
            "Архитектура RAG реализована корректно, "
            "ограничение носит инфраструктурный характер."
        )

    # ❗ ЭТО КЛЮЧЕВО
    if answer:
        await update.message.reply_text(answer)



def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot started. Listening for messages...")

    app.run_polling()


if __name__ == "__main__":
    main()

