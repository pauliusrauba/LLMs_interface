# Redefining Digital Health Interfaces with Large Language Models

This repository provides the implementation of our paper, [Redefining Digital Health Interfaces with Large Language Models](https://arxiv.org/abs/2310.03560). Our work demonstrates how large language models (LLMs) can be effectively integrated with digital health tools using **LangChain**.

## Overview

The repository includes scripts and resources to replicate the experiments and results presented in the paper. It also provides an interactive interface using Streamlit to explore the functionality of a Retrieval-Augmented Generation (RAG)-based system.

## Prerequisites

To use this repository, you will need an API key for querying the LLM. The simplest way to set this up is to create a Python file named `openai_config.py` in the root directory with the following structure:

```python
def get_openai_config():
    openai_config = {
        "api_type": "azure",
        "api_base": api_base,
        "api_version": api_version,
        "api_key": api_key_main,
        "deployment_id": deployment_name,
        "deployment_id_ada": deployment_name_ada,
        "temperature": 0.0,
        "seed": 0
    }
    return openai_config
```
Replace the placeholders (api_base, api_version, api_key_main, deployment_name, deployment_name_ada) with your actual configuration values. 

## Getting started

### 1. Install dependencies
Ensure that you have Python 3.10 or later installed. Create a virtual environment and install the required packages by running:

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

### 2. Repository structure
Here is the structure of the repository.

```
.
├── README.md
├── app_cvd.py                # Streamlit-based interactive app
├── main_all_questions.py     # Script to replicate all questions from the paper
├── main_specific_question.py # Script to replicate a specific question
├── custom_tools.py           # Custom utilities for the project
├── model_utils.py            # Model-related utilities
├── openai_config.py          # Configuration file for LLM API keys (user-created)
├── resources/                # Folder containing data and documents
│   ├── documents/            # Supporting documents
│   ├── model/                # Pre-trained model files
│   └── patient_info/         # Synthetic patient information and background data
├── conversations/            # Conversation logs
└── requirements.txt          # Required Python packages
```

### 3. Running the project

- **Interactive interfaces**. To explore the RAG-based system interactively, launch the Streamlit app ```streamlit run app_cvd.py```. This will open a web-based interface where you can interact with the system.
- **Reproducing results from the paper**. To replicate results for all questions discussed in the paper, run ```python main_all_questions.py```.
- **Reproducing specific questions**. To reproduce specific questions, run ``python main_specific_question.py <index>``. Replace ```<index>``` with the index of the question you want to replicate.

### 4. Data description

- Synthetic background dataset: The file ```resources/patient_info/background_dataset.csv``` is a synthetically generated dataset used to compute SHAP values for the Autoprognosis2 model. This synthetic data replaces the original dataset, which is private. Substituting this file with the original data (if available) will yield accurate SHAP values.
- Patient information: The ```resources/patient_info/``` directory contains additional synthetic patient records used in the paper.

### 5. Citation

If you found this repo useful, please consider citing our paper:
```
@Article{imrie2023redefining,
    title={Redefining Digital Health Interfaces with Large Language Models}, 
    author={Fergus Imrie and Paulius Rauba and Mihaela van der Schaar},
    year={2023},
    journal={arXiv preprint arXiv:2310.03560}
}
```
