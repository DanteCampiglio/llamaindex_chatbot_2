FROM python:3.11-slim

# Evita archivos .pyc y habilita logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos sólo los archivos necesarios primero para aprovechar cache
COPY proyecto_cero/requirements.txt /app/proyecto_cero/requirements.txt

RUN pip install --no-cache-dir -r /app/proyecto_cero/requirements.txt

# Copia el código de la app
COPY proyecto_cero /app/proyecto_cero

# Asegura imports del paquete local
ENV PYTHONPATH=/app/proyecto_cero/src

# Directorio de datos (se recomienda montarlo como volumen)
VOLUME ["/app/proyecto_cero/data"]

# Comando por defecto: CLI interactivo
CMD ["python", "proyecto_cero/main.py"]

