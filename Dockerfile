FROM python:3.6

# EXPOSE 8080

COPY /app /app

COPY requirements.txt /
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r /requirements.txt

# CMD ["python", "app/app.py"]
EXPOSE 5000
WORKDIR /app
CMD ["flask", "run", "--port=5000", "--host=0.0.0.0"]