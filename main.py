import asyncio
import logging

from aiogram.enums import ParseMode
from aiohttp import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

from course_master import ConsultGPT, llm

from environs import Env

env = Env()
env.read_env(".env")
bot_token = env("BOT_TOKEN")

consult_agent = None


async def main():
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(token=bot_token, parse_mode=None)
    logging.basicConfig(level=logging.INFO)

    @dp.message(Command(commands=["start"]))
    async def command_start(message):
        global consult_agent
        consult_agent = ConsultGPT.from_llm(llm, verbose=False)
        consult_agent.seed_agent()
        ai_message = consult_agent.ai_step()
        await message.answer(ai_message)

    @dp.message(F.text)
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

    @dp.message(~F.text)
    async def process_other(message):
        await message.answer("Бот принимает только текстовые сообщения")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
