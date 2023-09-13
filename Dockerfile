FROM python:3.8.5

WORKDIR /home/lanwin/MediVerse

COPY . .

RUN apt-get update && apt-get install -y python3-pip

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8529

CMD ["streamlit", "run", "app.py", "--server.port", "8529"]
