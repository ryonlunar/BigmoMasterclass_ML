import numpy as np
import matplotlib.pyplot as plt


def plot_weight_distribution(model, layer_indices=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layer_list)))

    n = len(layer_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        layer = model.layer_list[idx]
        weights = layer.W.flatten() if layer.W is not None else np.array([])
        ax.hist(weights, bins=30, color='steelblue', edgecolor='white')
        ax.set_title(f"Layer {idx} - Weights")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_gradient_distribution(model, layer_indices=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layer_list)))

    n = len(layer_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        layer = model.layer_list[idx]
        grads = layer.dW.flatten() if layer.dW is not None else np.array([])
        ax.hist(grads, bins=30, color='coral', edgecolor='white')
        ax.set_title(f"Layer {idx} - Weight Gradients")
        ax.set_xlabel("Gradient value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
