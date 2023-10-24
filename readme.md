# Fine-Tune quantized LLAMA2-7B model using QLoRA for Code Summarization
This repo provides code to fine-tune LLAMA2-7B by performing 4-bit quantization the model weights and use QLoRA to update the weights for Code Summarization task with [CodeSearchNet](https://arxiv.org/abs/1909.09436) dataset.

In my experiments the model was trained on NVIDIA A100 GPU - 40GB in Google Colab Platform. 

Note: The model was fine-tuned on the `python` subset of the dataset.

## Dependencies
- pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
- pip install datasets
- pip install bitsandbytes
- pip install einops
- pip install huggingface_hub
- pip install pytorch-ignite

## Dataset
The dataset used is the [CodeSearchNet](https://arxiv.org/abs/1909.09436), which comprises 6 million functions from open source code. It is divided into six languages: `python`, `java`, `javascript`, `php`, `ruby` and `go`. Furthermore, the dataset is split into three parts: `train`, `valid` and `test`. The dataset is available in the `JSONL` format.


## Dataset Download
You can download dataset from the google drive [link](https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h), or use the following command.

```bash
pip install gdown
mkdir data data/code2nl
cd data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../..
```

## Pre-processing the dataset
The dataset is pre-processed using the `data_preprocessing.ipynb` notebook. It reads the original dataset in `jsonl` file and constructs prompts for the model to train, and saves the data along with prompts as a `csv` file. The `text` column (prompts) in the `csv` file is then used to train the model.

We construct prompts for LLAMA-2, a Decoder-Only architecture, using continuous text data to fine-tune and update model weights. Unlike Encoder-Decoder architecture, source code cannot be used as input and target code-summary as output.

## Fine-tuning the model
To train the model, with the hyper-parameters used in the code, run the cells on an instance with A100 GPU. The model will be saved in the `results` directory.

Before running the cells, make sure to upload the dataset from pre-processing step in `csv` format to the HuggingFace dataset instance. Also, change the `dataset_name` variable to the dataset name in HuggingFace.

At the notebook's end, the model is uploaded to the HuggingFace model hub, where it can be downloaded for inference.

The model hyper-parameters can be adjusted to experiment with various configurations, depending on the available hardware.

## Flexibility
You can fine-tune various models, such as Mistral AI, by modifying the model_name variable in the code. Choose any model from the HuggingFace model hub, and it will be downloaded and fine-tuned on your dataset.