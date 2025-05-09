import datasets
import transformers
import torch
import os

import sys
from pathlib import Path

def main():

    root_dir = os.path.dirname(os.path.abspath(__file__))
    folder_A = os.path.join(root_dir, "A")
    folder_B = os.path.join(root_dir, "B")
    sys.path.append(folder_A)
    sys.path.append(folder_B)

    from rapper_classification import rapper_classifier
    from lyrics_generator import lyrics_generator
    from lyrics_generator_model import demo

    print("Running rapper classification...")
    rapper_classifier()

    print("Running lyrics generator...")
    lyrics_generator()

    print("Running demo...")
    demo()

if __name__ == "__main__":
    main()