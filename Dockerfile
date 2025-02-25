# 1️⃣ Python 3.9 tabanlı bir Docker image kullan
FROM python:3.9

# 2️⃣ Çalışma dizinini oluştur
WORKDIR /app

# 3️⃣ Gereksinim dosyasını kopyala
COPY requirements.txt .

# 4️⃣ Gerekli Python kütüphanelerini yükle
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Uygulama dosyalarını kopyala
COPY . .

# 6️⃣ FastAPI uygulamasını başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
