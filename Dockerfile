FROM nvidia/cuda:11.3.1-cudnn8-runtime

WORKDIR /pearl-pgm

COPY README.md ./
COPY scripts/publish.sh ./

RUN apt-get update \
	&& apt-get install -y python3 python3-pip python-is-python3

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml ./
COPY poetry.lock ./

RUN poetry install --no-root -vvv

COPY pearl ./pearl
COPY tests ./tests

RUN poetry install -vvv

CMD bash
