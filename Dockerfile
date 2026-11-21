FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home appuser
COPY app ./app
COPY artifacts ./artifacts
USER appuser
EXPOSE 5000
CMD ["python", "-m", "flask", "--app", "app.service", "run", "--host", "0.0.0.0", "--port", "5000"]
