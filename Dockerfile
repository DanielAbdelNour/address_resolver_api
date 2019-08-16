FROM python:3.6

ENV LISTEN_PORT=80
EXPOSE 80

COPY /app /app

COPY requirements.txt /
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r /requirements.txt

CMD ["python", "app/main.py"]