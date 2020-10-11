#encoding=utf-8

import sys
import os

import tensorflow as tf

from data_loader import census_dataset
from model import lr

lr_input_fn = census_dataset.input_fn
lr_model_fn = lr.model_fn

params = {
    "corpus_size": 267,
    "batch_size": 100,
    "learning_rate": 0.01,
    "train_data_path": "./data/census_data/adult.data.encoded",
    "test_data_path": "./data/census_data/adult.test.encoded",
}

tf.logging.set_verbosity(tf.logging.INFO)
run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=1000,
    log_step_count_steps=1000)
estimator = tf.estimator.Estimator(lr_model_fn, model_dir="checkpoint", config=run_config, params=params)

for i in range(3):
    estimator.train(input_fn=lambda: lr_input_fn(params["train_data_path"], 2, True, params["batch_size"]), max_steps=10000)
    estimator.evaluate(input_fn=lambda: lr_input_fn(params["test_data_path"], 1, False, 16281))
