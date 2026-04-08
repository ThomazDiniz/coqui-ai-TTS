ARG BASE=nvidia/cuda:12.8.1-base-ubuntu24.04
FROM ${BASE}
ARG BASE

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    gcc g++ make python3 python3-dev \
    espeak-ng libsndfile1-dev libc-dev ffmpeg && \
  rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/
ENV UV_NO_CACHE=1

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# tts-server and other non-interactive entrypoints cannot prompt for CPML; set to 0 to require a TTY / manual agreement.
ENV COQUI_TOS_AGREED=1

# Install dependencies first for better caching.
# PyTorch is optional in pyproject (extras "cpu" / "cuda"); without one of them, torch is never installed.
# With torch>=2.9, "codec-cuda" / "codec" provides torchcodec (required at import in TTS/__init__.py).
COPY pyproject.toml /app
RUN if echo "$BASE" | grep -q "cuda"; then \
      uv pip install -r pyproject.toml \
        --extra all --extra xtts_ft --extra cuda --extra codec-cuda \
        --torch-backend=cu128; \
    else \
      uv pip install -r pyproject.toml \
        --extra all --extra xtts_ft --extra cpu --extra codec \
        --torch-backend=cpu; \
    fi

# Copy the rest of the application
COPY . /app

# Install the project with the same torch-related extras as above
RUN if echo "$BASE" | grep -q "cuda"; then \
      uv pip install -e ".[cuda,xtts_ft,codec-cuda]"; \
    else \
      uv pip install -e ".[cpu,xtts_ft,codec]"; \
    fi

ENTRYPOINT ["tts"]
CMD ["--help"]
