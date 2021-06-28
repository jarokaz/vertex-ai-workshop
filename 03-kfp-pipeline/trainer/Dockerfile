
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-4

WORKDIR /trainer

# Copies the trainer code to the docker image.
COPY train.py .

ENTRYPOINT ["python", "train.py"]
