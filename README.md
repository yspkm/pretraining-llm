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
   pip3 install --upgrade pip wheel setuptools
   pip3 install -r requirements.txt
   pip3 install torch torchinfo --extra-index-url https://download.pytorch.org/whl/cu121
   ```

## Usage

### Data Preprocessing
To preprocess the data, run the following command:
```bash
python ./data/prep_dataset.py
```
This script tokenizes the dataset using `tiktoken` for speedy encoding and saves the tokenized data into binary files for training, validation, and evaluation.

### Training the Model
To train the model, run the following command:
```bash
python main.py
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
  dim_model: 2048 
  dim_hidden: 6144 
  num_heads: 32
  prob_dropout: 0.50
  batch_size: 20 
  val_interval: 1024 
  total_steps: 550000
  val_steps: 50 
  grad_accum_steps: 25
  lr_peak: 0.00005
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