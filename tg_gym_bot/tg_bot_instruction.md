# Инструкция по запуску Telegram бота

1. Откройте терминал и выполните:
```     
git clone https://github.com/MoshPit123/My_ML_Projects.git
 
cd My_ML_Projects/tg_gym_bot
```
   
2. Создайте виртуальное окружение и установите библиотеки:
```
python3 -m venv venv
   
source venv/bin/activate

pip install -r requirements.txt
```
3. Перед запуском задайте переменные окружения:
```
export TELEGRAM_BOT_TOKEN="ваш_токен_бота"

export HUGGINGFACE_API_TOKEN="ваш_токен_huggingface"
```
4. Запуск и остановка бота:

Запуск:
```
python tg_gym_bot.py
```
Остановка: 
```
Ctrl + C
```
