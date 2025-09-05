# Dynamic Batch Size Loader

A lightweight utility for dynamically changing batch sizes during
training in **PyTorch** and **TensorFlow**.

Instead of sticking to a fixed batch size, this tool allows you to
**adjust batch size based on training progress** (e.g., every 20% of the
dataset/epochs). This approach can stabilize training early on and
accelerate convergence later.

------------------------------------------------------------------------

## 🚀 Features

-   Framework-agnostic: works with **PyTorch** and **TensorFlow**\
-   Define **percentage-based intervals** for switching batch sizes\
-   Easy-to-use class-based structure\
-   Helps balance **training stability** and **training speed**

------------------------------------------------------------------------

## 📦 Installation

Clone the repo:

``` bash
git clone https://github.com/FrednadFari/DynamicBatchSizeLoader.git
cd DynamicBatchSizeLoader
```

No extra dependencies required beyond PyTorch / TensorFlow.

------------------------------------------------------------------------

## 📖 Usage

### 1. Import and initialize

``` python
from dynamic_batch_loader import DynamicBatchSizeLoader

loader = DynamicBatchSizeLoader(framework='pytorch')  # or 'tensorflow'
```

### 2. Define batch size strategy

``` python
loader.set_batch_sizes(
    batch_sizes=[32, 64, 128, 256, 512],
    percent_intervals=[20, 40, 60, 80, 100]
)
```

This means:\
- 0--20% → batch size 32\
- 20--40% → batch size 64\
- 40--60% → batch size 128\
- 60--80% → batch size 256\
- 80--100% → batch size 512

### 3. Use in your training loop

``` python
for progress, batch in enumerate(dataset):
    bs = loader.get_batch_size_for_progress(progress / len(dataset) * 100)
    # apply bs to your DataLoader / tf.data pipeline
```

------------------------------------------------------------------------

## 💡 Why Dynamic Batch Sizes?

-   **Small batch sizes** in the beginning → better generalization,
    smoother gradients.\
-   **Larger batch sizes** later → faster training and better GPU
    utilization.\
-   Flexible strategy for different datasets and model sizes.

------------------------------------------------------------------------

## 🛠️ Roadmap

-   [ ] Add automatic scheduler (increase batch size every `n` epochs)\
-   [ ] Add support for curriculum learning extensions\
-   [ ] Benchmark performance across popular datasets (CIFAR-10, MNIST,
    etc.)

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.

------------------------------------------------------------------------

👉 If you like this idea, don't forget to ⭐ the repo!
