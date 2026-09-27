"""Download the configured local ranking models into the persistent data volume."""
from app.local_ranking import prepare_models


def main():
    config = prepare_models()
    print(f"CrossEncoder cached: {config.model}; MMR embeddings: {config.embedding_backend}")
    if config.embedding_backend == "local":
        print(f"Candidate embedding model cached: {config.embedding_model}")


if __name__ == "__main__":
    main()
