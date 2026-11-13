from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image


class FakeSVDataset(Dataset):
    def __init__(self):
        super(FakeSVDataset, self).__init__()
    
    def _get_complete_data(self):
        # Look for data_complete.jsonl in the shared ExMRD data directory
        candidates = [
            r'F:\Bishe_workspace\ExMRD-master\data\FakeSV\data.jsonl',
            r'F:\Bishe_workspace\fakesv-qwen3\data.json',
            './data/FakeSV/data.jsonl',
        ]
        path = next((p for p in candidates if __import__('os').path.exists(p)), candidates[0])
        data_complete = pd.read_json(path, orient='records', dtype=False, lines=True)
        replace_values = {'辟谣': 2, '假': 1, '真':0}
        data_complete['label'] = data_complete['annotation'].replace(replace_values)
        data_complete = data_complete[data_complete['label']!=2]
        data_complete['event'], _ = pd.factorize(data_complete['keywords'])
        return data_complete
    
    def _get_data(self, fold, split):
        if fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        elif fold in ['temporal']:
            data = self._get_temporal_data(split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        _base = r'F:\Bishe_workspace\ExMRD-master\data\FakeSV\vids'
        if split == 'train':
            vid_path = f'{_base}/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'{_base}/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        # New 70/15/15 temporal split without 辟谣 videos
        # split values: 'train' / 'valid' / 'test'  ->  train_temporal / val_temporal / test_temporal
        split_name = 'val' if split == 'valid' else split
        vid_path = rf'F:\Bishe_workspace\ExMRD-master\data\FakeSV\vids\{split_name}_temporal.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data


class FakeTTDataset(Dataset):
    def __init__(self):
        super(FakeTTDataset, self).__init__()
    
    def _get_complete_data(self):
        candidates = [
            'data/FakeTT/data.jsonl',
            r'F:\Bishe_workspace\ExMRD-master\data\FakeTT\data.jsonl',
            r'F:\Bishe_workspace\FakeTT\data.jsonl',
            r'F:\Bishe_workspace\FakeTT\data.json',   # actual location (JSONL format)
        ]
        path = next((p for p in candidates if __import__('os').path.exists(p)), candidates[0])
        data = pd.read_json(path, orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].map(replace_values).fillna(0).astype(int)
        data['event'], _ = pd.factorize(data['event'])
        # set type of video_id to str
        return data
    
    def _get_data(self, fold, split):
        if fold in ['temporal']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        _base = r'F:\Bishe_workspace\ExMRD-master\data\FakeTT\vids'
        if split == 'train':
            vid_path = f'{_base}/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'{_base}/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = rf'F:\Bishe_workspace\ExMRD-master\data\FakeTT\vids\vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data
    
class FVCDataset(Dataset):
    def __init__(self):
        super(FVCDataset, self).__init__()
    
    def _get_complete_data(self):
        data = pd.read_json('data/FVC/data.jsonl', orient='records', lines=True, dtype={'vid': 'str'})
        data = data[data['label'].isin(['fake', 'real'])]
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['label'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event_id'])
        data['video_id'] = data['vid']
        return data
    
    def _get_data(self, fold, split):
        if fold in ['temporal']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FVC/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FVC/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'data/FVC/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data


class WildFakeDataset(Dataset):
    """WildFake Chinese subset (1826 videos, pre-split into train/val/test_zh.jsonl)."""

    SPLIT_FILES = {
        "train": r"F:\Bishe_workspace\WildFake\train_zh.jsonl",
        "val":   r"F:\Bishe_workspace\WildFake\val_zh.jsonl",
        "test":  r"F:\Bishe_workspace\WildFake\test_zh.jsonl",
    }
    ALL_DATA_FILE = r"F:\Bishe_workspace\WildFake\data_zh.jsonl"

    def __init__(self):
        super(WildFakeDataset, self).__init__()

    def _get_complete_data(self):
        import json, os
        rows = []
        with open(self.ALL_DATA_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        data = pd.DataFrame(rows)
        data["video_id"] = data["video_id"].astype(str)
        data["label"] = data["label"].astype(int)
        return data

    def _get_data(self, fold, split):
        if fold == "temporal":
            return self._get_temporal_data(split)
        raise NotImplementedError(f"WildFake only supports fold='temporal', got: {fold}")

    def _get_temporal_data(self, split: str):
        import json
        # accept "valid" as alias for "val" (to match train.py convention)
        split_key = "val" if split == "valid" else split
        path = self.SPLIT_FILES.get(split_key)
        if path is None:
            raise ValueError(f"Invalid split '{split}', expected train/val/test")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        vids = {r["video_id"] for r in rows}
        data = self._get_complete_data()
        data = data[data["video_id"].isin(vids)]
        return data
