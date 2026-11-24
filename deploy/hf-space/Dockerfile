# Kelpod sandbox API on a Hugging Face Docker Space (free CPU tier).
# Installs Kelpod from GitHub at a pinned commit; bump KELPOD_REF and restart the Space to update.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 KELPOD_JOB_TTL_S=900 KELPOD_MAX_UPLOAD_MB=25 KELPOD_MAX_CONCURRENT=1
WORKDIR /home/user/app
ARG KELPOD_REF=__REF__
RUN pip install --no-cache-dir --user "kelpod[api] @ git+https://github.com/kevnantony/kelpod@${KELPOD_REF}"
COPY --chown=user app.py .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
