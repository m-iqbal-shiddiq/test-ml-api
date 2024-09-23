FROM python:3.10.13
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]