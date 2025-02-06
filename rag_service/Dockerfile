FROM python:3.12.4

COPY ./ /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt \
    && rm -f requirements.txt \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "app.py"]