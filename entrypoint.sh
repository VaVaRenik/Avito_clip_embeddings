#!/bin/sh
set -e

if [ ! -d "/app/data/train_images/avito-images-1-3" ] || [ -z "$(ls -A /app/data/train_images/avito-images-1-3)" ]; then
  echo ">>> Загружаем тренировочные изображения..."
  kaggle datasets download -d vanyarekunov/avito-images-1-3 -p /app/data/train_images
  unzip /app/data/train_images/avito-images-1-3.zip -d /app/data/train_images/avito-images-1-3
  rm /app/data/train_images/avito-images-1-3.zip
else
  echo ">>> Тренировочные изображения уже скачаны, пропускаем."
fi

if [ ! -d "/app/data/tables/avito-tables" ] || [ -z "$(ls -A /app/data/tables/avito-tables)" ]; then
  echo ">>> Загружаем таблицы..."
  kaggle datasets download -d vanyarekunov/avito-tables -p /app/data/tables
  unzip /app/data/tables/avito-tables.zip -d /app/data/tables/avito-tables
  rm /app/data/tables/avito-tables.zip
else
  echo ">>> Таблицы уже скачаны, пропускаем."
fi

exec python main.py
