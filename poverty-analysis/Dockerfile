FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 python3 pip
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . ./

ADD app_poverty_analysis.py /

EXPOSE 8080
CMD gunicorn -b 0.0.0.0:8080 app_poverty_analysis:server
