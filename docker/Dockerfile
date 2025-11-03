# Базовий образ Python
FROM python:3.10-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо все
COPY . /app

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Вказуємо, як запускати бота
CMD ["python", "app/app.py"]

