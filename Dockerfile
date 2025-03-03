FROM tensorflow/serving:latest

COPY ./serving_model_dir /models
ENV MODEL_NAME=Fake_News_football_model