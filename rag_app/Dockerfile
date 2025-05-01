FROM python:3.11-slim
# install the udpate
RUN apt-get update
# install the packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# working dir
WORKDIR /app
# entry points
CMD ["tail", "-f", "/dev/null"]