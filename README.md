# Redefining Digital Health Interfaces with Large Language Models

This repository contains our implementation of [Redefining Digital Health Interfaces with Large Language Models](https://arxiv.org/abs/2310.03560).

We illustrate how LLMs can use digital health tools, implemented using langchain.

To get this up and running:

1. Set up an OpenAI api key and add it to apikey.py as "apikey"
2. Open streamlit via "streamlit run app_cvd.py"

Note: you don't really need to upload "person.csv" on streamlit, internally the function simply reads the data from the local directory.

If you found this repo useful, please consider citing our paper:
```
@Article{imrie2023redefining,
    title={Redefining Digital Health Interfaces with Large Language Models}, 
    author={Fergus Imrie and Paulius Rauba and Mihaela van der Schaar},
    year={2023},
    journal={arXiv preprint arXiv:2310.03560}
}
```
