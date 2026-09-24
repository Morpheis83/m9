# ============================================================
# IMAGE DE BASE
# ============================================================

FROM python:3.13-slim


# ============================================================
# CONFIGURATION PYTHON
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# ============================================================
# DEPENDANCES
# ============================================================

# Copier les dépendances avant le code permet de conserver
# le cache Docker lorsque seules les sources sont modifiées.
COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt


# ============================================================
# CODE SOURCE
# ============================================================

COPY src/ ./src/


# ============================================================
# UTILISATEUR NON PRIVILEGIE
# ============================================================

RUN useradd \
    --uid 10001 \
    --create-home \
    --shell /bin/bash \
    appuser \
    && mkdir -p \
        /app/model \
        /app/runtime \
    && chown -R \
        appuser:appuser \
        /app

USER appuser


# ============================================================
# SERVICE FASTAPI
# ============================================================

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" \
    || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]