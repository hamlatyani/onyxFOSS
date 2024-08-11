import os


MODEL_SERVER_HOST = os.environ.get("MODEL_SERVER_HOST") or "localhost"
MODEL_SERVER_ALLOWED_HOST = os.environ.get("MODEL_SERVER_HOST") or "0.0.0.0"
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT") or "9000")
# Model server for indexing should use a separate one to not allow indexing to introduce delay
# for inference
INDEXING_MODEL_SERVER_HOST = (
    os.environ.get("INDEXING_MODEL_SERVER_HOST") or MODEL_SERVER_HOST
)
INDEXING_MODEL_SERVER_PORT = int(
    os.environ.get("INDEXING_MODEL_SERVER_PORT") or MODEL_SERVER_PORT
)

# Danswer custom Deep Learning Models
INTENT_MODEL_VERSION = "danswer/hybrid-intent-token-classifier"
INTENT_MODEL_TAG = "v1.0.3"

# Bi-Encoder, other details
DOC_EMBEDDING_CONTEXT_SIZE = 512

# Cross Encoder Settings
ENABLE_RERANKING_ASYNC_FLOW = (
    os.environ.get("ENABLE_RERANKING_ASYNC_FLOW", "").lower() == "true"
)
ENABLE_RERANKING_REAL_TIME_FLOW = (
    os.environ.get("ENABLE_RERANKING_REAL_TIME_FLOW", "").lower() == "true"
)

# Used for loading defaults for automatic deployments and dev flows
DEFAULT_CROSS_ENCODER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-xsmall-v1"
DEFAULT_CROSS_ENCODER_API_KEY = os.environ.get("DEFAULT_CROSS_ENCODER_API_KEY")

# This controls the minimum number of pytorch "threads" to allocate to the embedding
# model. If torch finds more threads on its own, this value is not used.
MIN_THREADS_ML_MODELS = int(os.environ.get("MIN_THREADS_ML_MODELS") or 1)

# Model server that has indexing only set will throw exception if used for reranking
# or intent classification
INDEXING_ONLY = os.environ.get("INDEXING_ONLY", "").lower() == "true"

# notset, debug, info, warning, error, or critical
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
