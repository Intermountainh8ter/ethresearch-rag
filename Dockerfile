# Minimal runtime image: numpy is the only dependency. Index is built at image build time so
# `docker run -p 8080:8080 ethrag` answers immediately.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . \
 && python -m ethrag.cli build-index --data data --out /app/.index \
 && rm -rf data

ENV ETHRAG_INDEX=/app/.index
EXPOSE 8080
ENTRYPOINT ["ethrag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
