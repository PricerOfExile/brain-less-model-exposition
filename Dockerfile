FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry

COPY . /app

RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

CMD ["./.venv/bin/python3", "generic_pytorch_model/main.py"]