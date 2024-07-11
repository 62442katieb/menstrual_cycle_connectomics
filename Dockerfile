FROM ubuntu:22.04

WORKDIR /home

RUN mkdir /home/IDConn && mkdir /home/code
RUN apt-get update && apt-get install -y python3.9 python3-pip 

COPY Projects/IDConn/IDConn /home/IDConn
COPY Data/hormoneXfc /home/

WORKDIR /home/IDConn
RUN pip install . #&& pip install jupyter -U && pip install jupyterlab
WORKDIR /

#EXPOSE 8888

#ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
CMD python3 /home/code/nbs_predict-bc.py