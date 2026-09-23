# ============================================================
# Image Python légère
# ============================================================

FROM python:3.13-slim


# ============================================================
# Configuration Python
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ============================================================
# Répertoire de travail
# ============================================================

WORKDIR /app


# ============================================================
# Dépendances Python
#
# On copie requirements.txt avant le code pour profiter
# du cache Docker lorsque seules les sources changent.
# ============================================================

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt


# ============================================================
# Copie du code source
# ============================================================

COPY src/ ./src/

# ============================================================
# Utilisateur non privilégié
# ============================================================

RUN useradd \
    --uid 10001 \
    --create-home \
    --shell /bin/bash \
    appuser \
    && chown -R appuser:appuser /app


# ============================================================
# Répertoire de logs
# ============================================================

RUN mkdir -p \
    /app/model \
    /app/runtime \
    && chown -R \
       10001:10001 \
       /app



USER appuser


# ============================================================
# Port FastAPI
# ============================================================

EXPOSE 8000


# ============================================================
# Healthcheck Docker
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" \
    || exit 1


# ============================================================
# Démarrage de l'API
# ============================================================

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]