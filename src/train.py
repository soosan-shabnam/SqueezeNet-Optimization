from src import config
from models.SqueezeNet import SqueezeNet

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import keras_tuner as kt
import numpy as np


def train(traingen, validgen, testgen):
    es = EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True)

    n_steps = traingen.samples
    n_val_steps = validgen.samples

    tuner = kt.BayesianOptimization(
        SqueezeNet,
        objective="val_accuracy",
        max_trials=10,
        seed=42,
        directory=config.OUTPUT_PATH)

    print("[INFO] performing hyperparameter search...")
    tuner.search(
        traingen,
        validation_data=validgen,
        batch_size=config.BATCH_SIZE,
        callbacks=[es],
        epochs=config.EPOCHS
    )

    bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(bestHP)

    print("[INFO] training the best model...")
    model = tuner.hypermodel.build(bestHP)
    H = model.fit(traingen,
                  validation_data=validgen, batch_size=config.BATCH_SIZE,
                  epochs=config.EPOCHS, callbacks=[es], steps_per_epoch=n_steps,
                  validation_steps=n_val_steps, verbose=1)

    true_classes = testgen.classes
    class_indices = traingen.class_indices
    class_indices = dict((v, k) for k, v in class_indices.items())

    squeezenet_preds = model.predict(testgen)
    squeezenet_pred_classes = np.argmax(squeezenet_preds, axis=1)

    squeezenet_acc = accuracy_score(true_classes, squeezenet_pred_classes)
    print("SqueezeNet Model Accuracy with Bayesian Optimization: {:.2f}%".format(squeezenet_acc * 100))

    print(classification_report(true_classes, squeezenet_pred_classes))

    print(confusion_matrix(true_classes, squeezenet_pred_classes))