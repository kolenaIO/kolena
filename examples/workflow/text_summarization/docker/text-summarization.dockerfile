FROM python:3.9-slim AS base

WORKDIR /opt/text_summarization/

FROM base AS builder

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
RUN python3 -m pip install poetry

COPY pyproject.toml ./
COPY text_summarization ./text_summarization
# use cpu-only pytorch
RUN poetry add https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.0.0%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl
RUN poetry install --only main

FROM base

COPY --from=builder /opt/text_summarization/.venv .venv/
COPY --from=builder /opt/text_summarization/text_summarization ./text_summarization/

ENTRYPOINT [ "/opt/text_summarization/.venv/bin/python", "text_summarization/remote_evaluator.py" ]
