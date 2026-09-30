# Hugging Face Spaces (Docker SDK). See:
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# HF Spaces runs containers as UID 1000 -- create a matching user so
# file ownership/permissions inside the container are correct.
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install dependencies first so this layer is cached across rebuilds
# that only change application code.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app.
COPY --chown=user . .

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/app/.cache/huggingface

# HF Spaces expects the app on port 7860 (see README.md's app_port).
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
