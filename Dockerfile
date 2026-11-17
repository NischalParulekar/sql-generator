# The TableQA service. TAPAS is left out by default: it adds PyTorch and a 443 MB model.
FROM python:3.11.10-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
ARG EXTRAS="--extra service --extra agent"
RUN uv sync --locked --no-dev $EXTRAS

# The WTQ release supplies the demo's sample tables; fetch it once, at build time.
RUN uv run --no-sync python -m tableqa.data.wtq

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "tableqa.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
