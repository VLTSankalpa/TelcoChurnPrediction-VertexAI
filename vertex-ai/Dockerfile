
FROM asia-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-6:latest

WORKDIR /

# Copies the trainer code to the Docker image.
COPY ../vertex-ai/tune.py /trainer
COPY ../vertex-ai/hptuning_config.yaml /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.tune"]
