FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QIWEN_DATA_DIR=/app/.data

WORKDIR /app/backend
COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt

ARG INSTALL_KNOWLEDGE=false
RUN if [ "$INSTALL_KNOWLEDGE" = "true" ]; then \
      python -m pip install --no-cache-dir 'chromadb>=1,<2'; \
    fi

COPY . ./
RUN groupadd --system qiwen && useradd --system --gid qiwen --home-dir /app qiwen \
    && mkdir -p /app/.data && chown -R qiwen:qiwen /app

USER qiwen
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
