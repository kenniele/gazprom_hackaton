import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

from course_master import ConsultGPT, llm


bot_token = "6863497960:AAF5cO6t8uErM54jpuxV7zF1UNQtlqPMeIA"

consult_agent = None


async def main():
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(token=bot_token, parse_mode=None)
    logging.basicConfig(level=logging.INFO)

    @dp.channel_post(Command(commands=["start"]))
    async def command_start(message):
        global consult_agent
        consult_agent = ConsultGPT.from_llm(llm, verbose=False)
        consult_agent.seed_agent()
        ai_message = consult_agent.ai_step()
        await message.answer(ai_message)

    @dp.channel_post(F.text)
    async def process_text(message):
        if consult_agent is None:
            await message.answer("Используйте команду /start")
        else:
            human_message = message.text
            if human_message:
                consult_agent.human_step(human_message)
                consult_agent.analyse_stage()
            ai_message = consult_agent.ai_step()
            await message.answer(ai_message)

    @dp.channel_post(~F.text)
    async def process_other(message):
        await message.answer("Бот принимает только текстовые сообщения")

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=["channel_post"])


if __name__ == "__main__":
    asyncio.run(main())
