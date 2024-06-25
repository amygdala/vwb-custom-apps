FROM mintplexlabs/anythingllm:latest

USER root

ENV STORAGE_LOCATION=/app/server/storage
ENV STORAGE_DIR=/app/server/storage

COPY --chown=anythingllm:anythingllm ./allm_storage/ /app/server/storage/

ENV LLM_PROVIDER=anthropic
ENV DISABLE_TELEMETRY=true
ENV VECTOR_DB=lancedb
ENV EMBEDDING_ENGINE=native

RUN chown -R anythingllm:anythingllm /app/server/storage

# RUN mkdir -p /app/work
# RUN chown -R anythingllm:anythingllm /app/work


USER anythingllm
