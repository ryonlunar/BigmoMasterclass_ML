import numpy as np
from layers.layer import Layer
from losses import Losses


class FFNN:
    def __init__(self):
        self.layer_list: list[Layer] = []
        self.loss_func = None
        self.d_loss_func = None

    def add(self, layer: Layer):
        self.layer_list.append(layer)

    def compile(self, loss: str = 'mse'):
        if len(self.layer_list) == 0:
            raise ValueError("Model tidak memiliki layer")

        input_size = self.layer_list[0].input_dim
        if input_size is None:
            raise ValueError("Layer pertama harus memiliki input_dim")

        for layer in self.layer_list:
            layer.build(input_size)
            input_size = layer.units

        self.loss_func, self.d_loss_func = Losses.get(loss)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for layer in self.layer_list:
            output = layer.forward(output)
        return output

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            batch_size=32, learning_rate=0.01, epoch=100, verbose=1):
        history = {'train_loss': [], 'val_loss': []}
        n = X_train.shape[0]

        for ep in range(epoch):
            indices = np.random.permutation(n)
            X_sh = X_train[indices]
            y_sh = y_train[indices]

            batch_losses = []
            for start in range(0, n, batch_size):
                X_b = X_sh[start:start + batch_size]
                y_b = y_sh[start:start + batch_size]

                output = self.predict(X_b)
                batch_losses.append(self.loss_func(y_b, output))

                error = self.d_loss_func(y_b, output)
                for layer in reversed(self.layer_list):
                    error = layer.backward(error, learning_rate)

            train_loss = float(np.mean(batch_losses))
            history['train_loss'].append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.loss_func(y_val, val_pred)
                history['val_loss'].append(val_loss)

            if verbose == 1:
                msg = f"Epoch {ep + 1}/{epoch} - loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                print(msg)

        return history

    def save(self, path: str):
        from utils.persistence import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path: str):
        from utils.persistence import load_model
        return load_model(path)

    def plot_weights(self, layer_indices=None):
        from utils.visualization import plot_weight_distribution
        plot_weight_distribution(self, layer_indices)

    def plot_gradients(self, layer_indices=None):
        from utils.visualization import plot_gradient_distribution
        plot_gradient_distribution(self, layer_indices)
