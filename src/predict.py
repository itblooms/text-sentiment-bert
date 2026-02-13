from transformers import pipeline
import torch
from argparse import ArgumentParser


def make_prediction(model_path: str, text: str, device=torch.device("cuda")):
    pipe = pipeline("text-classification", model=model_path, device=device)
    return pipe(text)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type="str", required=True)
    parser.add_argument("--prompt", type="str", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(make_prediction(args.model_path, args.prompt, device))
    