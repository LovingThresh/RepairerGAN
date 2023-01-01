# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 11:45
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Optimizer.py
# @Software: PyCharm

import comet_ml
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("comet_ml")

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


def build_model_graph(experiment):
    model = Sequential()
    model.add(
        Dense(
            experiment.get_parameter("first_layer_units"),
            activation="sigmoid",
            input_shape=(784,),
        )
    )
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(),
        metrics=["accuracy"],
    )
    return model


def train(experiment, model, x_train, y_train, x_test, y_test):
    model.fit(
        x_train,
        y_train,
        batch_size=experiment.get_parameter("batch_size"),
        epochs=experiment.get_parameter("epochs"),
        validation_data=(x_test, y_test),
    )


def evaluate(experiment, model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    LOGGER.info("Score %s", score)


def get_dataset():
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


# Get the dataset:
x_train, y_train, x_test, y_test = get_dataset()

# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize MNIST Network",
    "spec": {"maxCombo": 10, "objective": "minimize", "metric": "loss"},
    "parameters": {
        "first_layer_units": {
            "type": "integer",
            "mu": 500,
            "sigma": 50,
            "scalingType": "normal",
        },
        "batch_size": {"type": "discrete", "values": [64, 128, 256]},
    },
    "trials": 1,
}

opt = comet_ml.Optimizer(config)

for experiment in opt.get_experiments(api_key="sDV9A5CkoqWZuJDeI9JbJMRvp", project_name="my_project", workspace="lovingthresh"):
    # Log parameters, or others:
    experiment.log_parameter("epochs", 10)

    # Build the model:
    model = build_model_graph(experiment)

    # Train it:
    train(experiment, model, x_train, y_train, x_test, y_test)

    # How well did it do?
    evaluate(experiment, model, x_test, y_test)

    # Optionally, end the experiment:
    experiment.end()
