FROM python:3.10-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Создание директорий для данных и результатов
WORKDIR /app
RUN mkdir -p data output

# Установка pip-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Настройка Kaggle API
ENV KAGGLE_CONFIG_DIR=/root/.kaggle
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Создание рабочей директории
WORKDIR /app

# Скачивание датасетов
RUN kaggle datasets download -d vanyarekunov/avito-images-1-3 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-1-patch-2 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-2-patch-1 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-2-patch-2 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-3-patch-1 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-3-patch-2 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-4-patch-1 -p /app/data --unzip && \
    # kaggle datasets download -d vavarenikk/avito-part-4-patch-2 -p /app/data --unzip && \
    # kaggle datasets download -d vanyarekunov/avito-test-patch-1 -p /app/data --unzip && \
    # kaggle datasets download -d vanyarekunov/avtio-test-patch-2 -p /app/data --unzip && \
    kaggle datasets download -d vanyarekunov/avito-tables -p /app/data --unzip

# Копирование твоего проекта (если нужно)
COPY . .

CMD ["python", "main.py"]
