# Dockerfile
FROM python:3.12-slim

# Встановлюємо всі необхідні build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо requirements і ставимо залежності
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install -r requirements.txt

# Копіюємо код
COPY . .

# Порт
ENV PORT=8080

EXPOSE 8080

# Запуск через Gunicorn
CMD ["gunicorn", "app:app", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:8080"]
