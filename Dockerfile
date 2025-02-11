FROM python:3.10

LABEL maintainer="krishna158@live.com"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_RUN_HOST 0.0.0.0
ENV ALSA_CONFIG_PATH=/etc/asound.conf

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        tk \
        alsa-utils \
        libasound2-dev\
        libsndfile1\
        wget

RUN apt-get install gcc -y

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["app.py"]