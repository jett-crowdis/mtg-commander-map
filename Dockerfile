# docker build -t jcrowdis/commander-map:2.2 .

FROM python:3.7.9
RUN apt-get update

WORKDIR /
CMD ["sh" , "-c", "command && bash"]

COPY scripts scripts
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel 
RUN python3 -m pip install Cython
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip --no-cache-dir install --upgrade awscli