# Вихідний базовий образ
FROM python:3.12

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Копіюємо файли requirements.txt для встановлення залежностей
COPY requirements.txt .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо всі файли проєкту в контейнер
COPY . .

# Експортуємо порт 5000 для Flask
EXPOSE 5000

# Команда для запуску додатку
CMD ["python", "app_final/app.py"]
