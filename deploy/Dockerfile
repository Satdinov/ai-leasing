FROM python:3.9-slim

WORKDIR /app

# Копируем всё содержимое проекта
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем PYTHONPATH
ENV PYTHONPATH="/app/app"

EXPOSE 8007

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]