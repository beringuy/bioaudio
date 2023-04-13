FROM python:3.10.6-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN export FLASK_APP='app.py'
RUN export FLASK_RUN_PORT=8080

EXPOSE 8080

CMD flask run --host 0.0.0.0 --port 8080