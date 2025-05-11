FROM python:3.10-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Создание директорий
RUN mkdir -p data/train_images data/test_images data/tables output

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Настройка Kaggle API
ENV KAGGLE_CONFIG_DIR=/root/.kaggle
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Скачивание и распаковка датасетов по категориям

# 🔽 Тренировочные изображения
RUN kaggle datasets download -d vanyarekunov/avito-images-1-3 -p /app/data/train_images && \
    unzip /app/data/train_images/avito-images-1-3.zip -d /app/data/train_images && \
    rm /app/data/train_images/avito-images-1-3.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-1-patch-2 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-1-patch-2.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-1-patch-2.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-2-patch-1 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-2-patch-1.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-2-patch-1.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-2-patch-2 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-2-patch-2.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-2-patch-2.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-3-patch-1 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-3-patch-1.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-3-patch-1.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-3-patch-2 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-3-patch-2.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-3-patch-2.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-4-patch-1 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-4-patch-1.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-4-patch-1.zip && \
#    kaggle datasets download -d vavarenikk/avito-part-4-patch-2 -p /app/data/train_images && \
#    unzip /app/data/train_images/avito-part-4-patch-2.zip -d /app/data/train_images && \
#    rm /app/data/train_images/avito-part-4-patch-2.zip

# Таблицы
RUN kaggle datasets download -d vanyarekunov/avito-tables -p /app/data/tables && \
    unzip /app/data/tables/avito-tables.zip -d /app/data/tables && \
    rm /app/data/tables/avito-tables.zip

# Тестовые изображения
#RUN kaggle datasets download -d vanyarekunov/avito-test-patch-1 -p /app/data/test_images && \
#    unzip /app/data/test_images/avito-test-patch-1.zip -d /app/data/test_images && \
#    rm /app/data/test_images/avito-test-patch-1.zip && \
#    kaggle datasets download -d vanyarekunov/avtio-test-patch-2 -p /app/data/test_images && \
#    unzip /app/data/test_images/avtio-test-patch-2.zip -d /app/data/test_images && \
#    rm /app/data/test_images/avtio-test-patch-2.zip \

# Копирование исходников проекта
COPY . .

# Точка входа
CMD ["python", "main.py"]

