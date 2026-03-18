# Bigmo Masterclass ML

Implementasi **Feedforward Neural Network (FFNN)** dari scratch menggunakan NumPy, tanpa framework deep learning. Mencakup forward propagation, backward propagation, berbagai fungsi aktivasi, loss function, weight initializer, serta layer normalisasi (RMSNorm) dan optimizer Adam.

## Fitur

- **Layer**: Dense (fully connected), RMSNorm
- **Aktivasi**: Linear, ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Swish
- **Loss**: MSE, Binary Cross-Entropy (BCE), Categorical Cross-Entropy (CCE)
- **Initializer**: Zero, Uniform, Random (Normal), Xavier Uniform, He
- **Optimizer**: SGD, Adam
- **Utilitas**: Save/load model (pickle), visualisasi distribusi weight & gradient

## Struktur Repository

```
BigmoMasterclass_ML/
├── notebook/
│   └── notebook.ipynb      # Eksplorasi data, training, dan evaluasi model
├── src/
│   ├── model.py            # Kelas FFNN (Sequential model)
│   ├── activations.py      # Fungsi aktivasi beserta turunannya
│   ├── losses.py           # Fungsi loss beserta turunannya
│   ├── initializers.py     # Fungsi inisialisasi bobot
│   ├── layers/
│   │   ├── layer.py        # Abstraksi kelas Layer
│   │   ├── dense.py        # Dense layer
│   │   └── rmsnorm.py      # RMSNorm layer
│   └── utils/
│       ├── persistence.py  # Save & load model
│       └── visualization.py# Plot distribusi weight & gradient
└── pyproject.toml
```

## Setup dan Cara Menjalankan

### Prasyarat

- Python >= 3.13
- [`uv`](https://github.com/astral-sh/uv) (package manager)

### Instalasi

```bash
uv sync
```

### Menjalankan Notebook

```bash
uv run jupyter notebook notebook/notebook.ipynb
```

### Aktivasi Virtual Environment (opsional)

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Dependensi

| Package | Versi |
|---|---|
| numpy | >= 2.4.2 |
| pandas | >= 3.0.1 |
| matplotlib | >= 3.10.8 |
| scikit-learn | >= 1.8.0 |
| jupyter | >= 1.1.1 |
| tqdm | >= 4.67.3 |

## Pembagian Tugas

| Nama | NIM | Tugas |
|---|---|---|
| Orvin Andika Ikhsan A | 13523017 | Fungsi aktivasi Sigmoid & Tanh, loss MSE & BCE, forward propagation, backward propagation |
| Fajar Kurniawan | 13523027 | Initializer zero, uniform, random; fungsi aktivasi Softmax, Leaky ReLU, Swish; loss CCE |
| Adhimas Aryo Bimo | 13523052 | Model FFNN, fungsi aktivasi ReLU & Linear, abstraksi kelas Layer, initializer He & Xavier, optimizer Adam |
