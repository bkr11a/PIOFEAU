__author__ = "Brad Rice"
__version__ = 0.1

import io

import tensorflow as tf
import matplotlib.pyplot as plt

class PredictOnEpochEndCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y

    def convertPlotToImage(self, figure):
        buffer = io.BytesIO()
        plt.savefig(buffer, format = 'png')
        plt.close(figure)
        buffer.seek(0)

        image = tf.image.decode_png(buffer.getvalue(), channels = 4)
        image = tf.expand_dims(image, 0)

        return image

    def plotResults(self, X, y, predictions, epoch):
        # domain = tf.linspace(0, 1, 100)

        # fig, axs = plt.subplots(3, 1, figsize = (20, 15))
        # ds = ["train", "validation", "test"]

        # for i in range(3):
            # profile = examples[i][0][0]
            # ax = axs[i]
            # ax.plot(domain, profile, label = f"Profile")
            # ax.plot(domain, solutions[i], "-.", label = f"Ground Truth")
            # ax.plot(domain, predictions[i], "-.", label = f"Predictions")
            # ax.set_title(f"Sample prediction on {ds[i]} dataset")
            # ax.set_xlabel('$t$', fontsize = 12)
            # ax.set_ylabel('$u(t)$', fontsize = 12)
            # ax.grid(True, color = 'lightgrey')
            # ax.legend()

        # plt.suptitle(f"Prediction Results after epoch {epoch}")

        # return fig
        pass

    def on_epoch_end(self, epoch, logs = None):
        predictions = []
        for x in self.X:
            prediction = self.model.predict(x, verbose = 0)
            predictions.append(prediction)

        fig = self.plotResults(self.X, self.y, predictions, epoch)
        img = self.convertPlotToImage(fig)

        # Write this out to the tensorboard
        tf.summary.image("Post Epoch Predictions", img, step = epoch)