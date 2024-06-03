# Pretraining Language Model

This repository contains the implementation of a language model training pipeline. The project is organized into different modules to facilitate ease of understanding and scalability.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to serve as a study for pretraining a large language model. The model is trained on a dataset provided in binary format. The implementation includes data processing, model definition, training loop, and logging.

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch
- Other dependencies listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pretraining-llm.git
   cd pretraining-llm
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt
   pip install torch torchinfo --extra-index-url https://download.pytorch.org/whl/cu121
   ```

## Usage

### Data Preprocessing
To preprocess the data, run the following command:
```bash
python main.py prep_data
```
This script tokenizes the dataset using `tiktoken` for speedy encoding and saves the tokenized data into binary files for training, validation, and evaluation.

### Training the Model
To train the model, run the following command:
```bash
python main.py train
```

### Running TensorBoard
To monitor the training process with TensorBoard, use the following command. The log directory is set to `rootdir/results/run_name`, which is generated based on the current configuration.

```bash
tensorboard --logdir rootdir/results/$(python -c "from logger import TrainingLogger; print(TrainingLogger.generate_run_name('AutoLLaMA', 'llama', {}, ''))") --port 6006 --host 0.0.0.0
```

### Configuration
The hyperparameters and other configurations are defined in the `config.yaml` file. You can modify this file to change the settings for training, such as batch size, learning rate, number of layers, etc.

## Project Structure
```
.
├── config.yaml           # Configuration file
├── data
│   ├── eval.bin          # Evaluation data
│   ├── prep_dataset.py   # Data preparation script
│   ├── train.bin         # Training data
│   └── val.bin           # Validation data
├── data.py               # Data loading and processing
├── etc
│   ├── main.py           # Entry point for training
│   └── trainer.py        # Training loop implementation
├── LICENSE               # License file
├── logger.py             # Logging and checkpointing
├── main.py               # Main script to start training
├── model.py              # Model definition
├── README.md             # This readme file
├── requirements.txt      # List of dependencies
└── trainer.py            # Training implementation
```

## Configuration
The `config.yaml` file contains all the configurations for the training process. Here is an example configuration:

```yaml
config:
  max_len_seq: 1024 
  num_layers: 32
  dim_model: 2304 
  dim_hidden: 9216 
  num_heads: 36
  prob_dropout: 0.50
  batch_size: 20 
  val_interval: 1000 
  total_steps: 550000
  val_steps: 50 
  grad_accum_steps: 25
  lr_peak: 0.000025 # 휴리스틱
  weight_decay: 0.01
  warmup_steps: 2000
  balance: [11, 11, 11, 1]
  devices: ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
wandb:
  project_name: "AutoLLaMA"
  model_name: "llama"
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.