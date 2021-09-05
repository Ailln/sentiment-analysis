import pandas as pd
from torch.utils.data import Dataset


class SADataset(Dataset):
    def __init__(self, tokenizer, conf):
        self.conf = conf
        self.df = pd.read_csv(self.conf["data"]["train_path"])
        self.tokenizer = tokenizer
        self.data_len = self.df.shape[0]

    def __getitem__(self, item):
        line = self.df.iloc[item]
        question = line["文本"]
        label = line["标签"]
        input_item = self.tokenizer(question, padding="max_length", truncation=True,
                                    max_length=self.conf["data"]["max_length"], return_tensors='pt')
        input_item = {k: v[0] for k, v in input_item.items()}
        input_item["labels"] = list(self.conf["data"]["vocab"].keys()).index(label)
        return input_item

    def __len__(self):
        return self.data_len
