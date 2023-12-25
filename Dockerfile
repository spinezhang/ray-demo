FROM python:3.11-slim-bookworm

RUN set -o errexit
RUN set -o nounset

RUN apt-get update && apt-get install -y gcc

RUN pip install pip --upgrade
RUN adduser pyrunner --home /home/pyrunner
USER pyrunner
WORKDIR /home/pyrunner

COPY --chown=pyrunner:pyrunner requirements.txt .
COPY --chown=pyrunner:pyrunner image_train_tensor_demo.py .
COPY --chown=pyrunner:pyrunner image_train_tensor.py .
COPY --chown=pyrunner:pyrunner image_train_raytensor.py .
COPY --chown=pyrunner:pyrunner data_builder ./data_builder
COPY --chown=pyrunner:pyrunner model ./model
COPY --chown=pyrunner:pyrunner dataset  ./dataset

RUN set -e && python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 

ENV PATH="/home/pyrunner/.local/bin:${PATH}"

EXPOSE 80

CMD [ "python", "./image_train_tensor_demo.py" ]
