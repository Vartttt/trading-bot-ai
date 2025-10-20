# Dockerfile
FROM python:3.12-slim

# Оптимізація pip
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=100

# Оновлення pip та setuptools
RUN pip install --upgrade pip setuptools wheel

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
RUN pip install -r requirements.txt

# Копіюємо увесь код
COPY . .

# Railway сам підставляє PORT (не фіксуй його)
EXPOSE 8080

# Запуск через Gunicorn (використовує PORT із середовища або 8080 за замовчуванням)
CMD gunicorn app:app --workers 2 --threads 4 --bind 0.0.0.0:${PORT:-8080}

