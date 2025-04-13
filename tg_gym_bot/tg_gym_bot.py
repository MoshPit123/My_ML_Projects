import asyncio
import logging
import nest_asyncio
from typing import List, Dict
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.methods import DeleteWebhook
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

TOKEN = 'Ваш сгенерированный токен Telegram-бота'
API_KEY = 'Bearer io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6ImY5YWMxNjFhLTI4MGYtNDQ5NC1hMzI0LTE3NDE4Y2QzMDUzMiIsImV4cCI6NDg5NjYwMjUxOX0.N12SLupj4u4E91ibiBzyem2eieMjCRbExcU8lWy06-9Oyy4zHcomuN9U3ca4J_Jbg7pLYyezYyM4TitVrSZtCA'
url = "https://api.intelligence.io.solutions/api/v1/chat/completions"

nest_asyncio.apply()

class RAG:
    def __init__(self, csv_path: str = "exercises.csv"):

        token =  "Ваш сгенерированный токен HuggingFace"

        # Инициализация модели эмбеддингов с токеном
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task = "feature-extraction",
            huggingfacehub_api_token=token,
            model_kwargs={'device': 'cpu'}
        )

        # Загрузка и подготовка данных
        self.vector_db = self._prepare_vector_db(csv_path)

        # Хранилище истории диалогов
        self.dialog_history: Dict[int, List[Dict[str, str]]] = {}

    def _prepare_vector_db(self, csv_path: str):
        """Создание векторной базы данных из CSV"""
        loader = CSVLoader(file_path=csv_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        return FAISS.from_documents(chunks, self.embedding_model)

    def _update_history(self, user_id: int, role: str, content: str):
        #Обновление истории диалога
        if user_id not in self.dialog_history:
            self.dialog_history[user_id] = []

        # Сохраняем последние 10 сообщений
        if len(self.dialog_history[user_id]) >= 10:
            self.dialog_history[user_id] = self.dialog_history[user_id][-9:]

        self.dialog_history[user_id].append({"role": role, "content": content})

    async def generate_answer(self, user_id: int, query: str) -> str:
        #Генерация ответа с использованием RAG и истории диалога
        # Обновляем историю новым сообщением пользователя
        self._update_history(user_id, "user", query)

        # Поиск релевантных чанков
        relevant_docs = self.vector_db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Формирование промпта для DeepSeek с историей диалога
        messages = [
            {
            "role": "system",
            "content": """
# SYSTEM PREAMBLE
1) You are an expert fitness trainer AI assistant with 10+ years of coaching experience. You specialize in creating personalized workout programs, nutrition plans, and providing exercise technique guidance.
2) Your knowledge covers strength training, cardio, mobility, rehabilitation, and sports-specific conditioning.
3) You must provide safe, scientifically-valid recommendations tailored to the user's fitness level and goals.
4) Always follow "Answering rules" strictly.

## ANSWERING RULES
1) When recommending exercises:
   - Include proper form instructions
   - Specify equipment requirements
   - List target muscle groups
   - Provide progression/regression options
2) For workout programs:
   - Structure by training days
   - Include sets/reps schemes
   - Specify rest periods
   - Offer modifications for different levels

## CHAIN OF THOUGHTS
1) **OBEY the EXECUTION MODE**
2) **REQUEST ANALYSIS:**
   - Identify user's fitness level (beginner/intermediate/advanced)
   - Determine training goals (strength, hypertrophy, endurance, etc.)
   - Note any injuries or limitations
3) **PROGRAM DESIGN:**
   - Select appropriate exercises from database
   - Structure workout split (e.g., push/pull/legs)
   - Determine optimal volume and intensity
4) **EXERCISE SELECTION:**
   - Choose exercises targeting required muscle groups
   - Include proper warm-up and cool-down
   - Provide alternatives for equipment availability
5) **SAFETY CHECK:**
   - Verify all recommendations are safe for user's level
   - Include technique cues to prevent injury
   - Suggest appropriate weight progression

## RESPONSE FORMAT
For exercise recommendations:


{Exercise Name} ({Equipment}) - {Target Muscles}
• Technique:
  1. {Step 1}
  2. {Step 2}
• Sets/Reps: {Recommendation}
• Progression: {Advanced variation}
• Regression: {Easier variation}


For workout programs:

{Day 1}: {Muscle Group/Focus}
1. {Exercise 1} 
   - {Sets}x{Reps}
   - {Rest period}
2. {Exercise 2}
   ...


## EXAMPLE TASK
User request: "Create a 3-day full body workout program for muscle building using dumbbells only"

AI response:

Request: Create a 3-day full body workout program for muscle building using dumbbells only.

Program Design:
- Training level: Intermediate
- Goal: Hypertrophy
- Equipment: Dumbbells only
- Rest between sets: 60-90 sec

Day 1: Upper Body Push
1. Dumbbell Bench Press (Dumbbells) - Chest, Triceps
   • Technique:
     1. Sit on a bench with a dumbbell in each hand, resting on your thighs
     2. Lean back and position the dumbbells to the sides of your chest, palms facing forward.
     3. Press the dumbbells upward until your arms are fully extended.
     4. Pause for a moment at the top, then slowly lower the dumbbells back to the starting position.
   • 4x8-10
   
2. Dumbbell Shoulder Press (...) 
...

Day 2: Lower Body
...

Day 3: Upper Body Pull
...


## SAFETY PROTOCOLS
- Never recommend exercises that could aggravate injuries
- Always include proper warm-up recommendations
- Provide clear technique instructions
- Specify when to consult a medical professional"""

            }
        ]

        # Добавляем историю диалога
        if user_id in self.dialog_history:
            messages.extend(self.dialog_history[user_id])

        # Добавляем текущий контекст и вопрос
        messages.append({
            "role": "user",
            "content": f"Контекст из базы знаний:\n{context}\n\nТекущий вопрос: {query}"
        })

        # Отправка запроса к Llama API
        response = requests.post(
            url,
            headers={
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "model": "SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B",
                "messages": messages,
                "temperature": 0.7
            }
        )

        answer = response.json()['choices'][0]['message']['content']

        # Добавляем ответ бота в историю
        self._update_history(user_id, "assistant", answer)

        return answer

# Инициализация бота
bot = Bot(TOKEN)
dp = Dispatcher()
rag = RAG()

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer("Привет! Я бот с интегрированной RAG-системой. Задайте вопрос.")

@dp.message()
async def handle_query(message: Message):
    try:
        answer = await rag.generate_answer(message.from_user.id, message.text)
        await message.answer(answer)
    except Exception as e:
        logging.error(f"Error: {e}")
        await message.answer("Произошла ошибка при обработке запроса")

async def main():
    await bot(DeleteWebhook(drop_pending_updates=True))
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())