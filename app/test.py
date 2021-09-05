from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch

from util.conf_util import get_default_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f">> use device: {device}")

conf = get_default_config()
print(conf)
bert_path = conf["train"]["pre_train_model"]  # https://huggingface.co/bert-base-chinese/tree/main
save_path = conf["train"]["save_path"]
vocab = conf["data"]["vocab"]
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=len(conf["data"]["vocab"]))
model.load_state_dict(torch.load(save_path), map_location=device)
print(">> load model success")
model.to(device)


def inference(data):
    token = tokenizer(data, padding="max_length", truncation=True, max_length=conf["data"]["max_length"],
                      return_tensors='pt').to(device)
    outputs = model(**token, labels=torch.LongTensor([1]).to(device))
    pred = torch.argmax(outputs.logits, dim=-1)
    return vocab[list(vocab.keys())[pred]]


if __name__ == '__main__':
    print("## 请输入文本")
    while True:
        data = input(">> ")
        if len(data) > 0:
            print(inference(data))
        else:
            print("## 文本为空，请重新输入")
