#!/usr/bin/env python3
import json
from lovelive_sdk import Dataset, Atom, DatasetType, BizType
import os
import tqdm


def make_lovelive_dataset(json_file, dataset_name):
    dataset = Dataset(dataset_name)
    dataset.commit(DatasetType.GroupAtom, biztype=BizType.Public)
    with open(json_file, 'r') as f:
        nids, file_dirs = json.load(f)
    for i in tqdm.tqdm(range(len(file_dirs))):
        name = os.path.basename(file_dirs[i])
        atom = Atom(name=name)
        atom.add_pics(nids[i*2])
        atom.add_pics(nids[i*2+1])
        atom.commit_to_dataset(dataset_name)


def main():
    datasets = {
        'train': [
            "./nori_json/train.json",
            "GoPro:train"
        ]
    }
    dataset = 'train'
    json_file, lovelive_dataset = datasets[dataset]
    make_lovelive_dataset(json_file, lovelive_dataset)


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
