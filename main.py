import datasets
import transformers
import torch

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import dataloader
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("clinc_oos", "small")

t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5_base")
t5_tokenizer = AutoTokenizer.from_pretrained("t5_base")
