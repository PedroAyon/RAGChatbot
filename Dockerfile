FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rag_chatbot ./rag_chatbot
COPY main.py .

# Non-root user; /app/output is where the bot writes charts and CSV exports.
RUN useradd --create-home appuser \
    && mkdir -p /app/output \
    && chown appuser /app/output
USER appuser

CMD ["python", "main.py"]