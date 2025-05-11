FROM python:3.10-slim

# 1) Системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
  && rm -rf /var/lib/apt/lists/*

# 2) Рабочая директория
WORKDIR /app

# 3) Копирование зависимостей и установка
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Настройка Kaggle API
ENV KAGGLE_CONFIG_DIR=/root/.kaggle
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# 5) Копируем код и entrypoint
COPY . .

# 6) Гарантируем, что точки монтирования существуют
RUN mkdir -p data/train_images data/test_images data/tables output

# 7) Устанавливаем entrypoint
ENTRYPOINT ["./entrypoint.sh"]
