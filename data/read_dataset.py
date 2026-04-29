import os
import torch
import numpy as np
import pickle
import gzip
from torch_geometric.data import Data

def read_dataset(path):
    dataset = []
    if path.endswith(".pt") or path.endswith(".pt.gz"):
        print(f"Loading torch dataset from {path} ...")
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rb") as f:
            dataset = torch.load(f, map_location="cpu", weights_only=False)

    elif os.path.isdir(path):
        npz_files = sorted([f for f in os.listdir(path) if f.endswith(".npz")])
        print(f"Loading {len(npz_files)} samples from folder ...")

        for fname in npz_files:
            file_path = os.path.join(path, fname)
            data_npz = np.load(file_path)

            edge_index = torch.tensor(data_npz["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(data_npz["edge_attr"], dtype=torch.float32)
            roi = torch.tensor(data_npz["roi"], dtype=torch.float32)
            label = int(data_npz["label"][0])
            sample_id = int(data_npz["sample_id"][0])

            x = roi  # Each node represented by an N-dimensional vector
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                sample_id=sample_id
            )
            dataset.append(data)

    else:
        raise ValueError(f"Invalid path or unsupported format: {path}")

    print(f"Finished loading {len(dataset)} graph samples.")
    return dataset


if __name__ == "__main__":
    path = "jacob_ABIDE_dataset.pt.gz"  # or "path/to/npz_folder"
    dataset = read_dataset(path=path)
