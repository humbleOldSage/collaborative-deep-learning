import sys

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    dataset_name = sys.argv[1]

    model = SentenceTransformer('allenai-specter')

    if dataset_name == 'citeulike-a':
        df = pd.read_csv(f'data/raw/{dataset_name}/raw-data.csv')
        content = df['title'] + ' ' + df['raw.abstract']

    elif dataset_name == 'citeulike-t':
        with open(f'data/raw/{dataset_name}/rawtext.dat') as f:
            lines = f.readlines()
        content = lines[1::2]
    else:
        df = pd.read_csv(f'data/raw/{dataset_name}/rawtext.csv')
        content = df['description']


    embeddings = model.encode(content, convert_to_tensor=True)
    print(embeddings.shape)
    torch.save(embeddings, f'data/processed/{dataset_name}/content-bert.pt')
