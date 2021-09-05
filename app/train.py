from transformers import AdamW
from transformers import get_scheduler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from datasets import load_metric
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from util.conf_util import get_default_config
from util.data_util import SADataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f">> use device: {device}")

conf = get_default_config()
print(conf)
batch_size = conf["train"]["batch_size"]
bert_path = conf["train"]["pre_train_model"]  # https://huggingface.co/bert-base-chinese/tree/main
learning_rate = conf["train"]["learning_rate"]
num_epochs = conf["train"]["num_epochs"]
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=len(conf["data"]["vocab"]))
model.to(device)

weight_name_list = [
    # "bert.encoder.layer.10.attention.self.query.weight",
    # "bert.encoder.layer.10.attention.self.query.bias",
    # "bert.encoder.layer.10.attention.self.key.weight",
    # "bert.encoder.layer.10.attention.self.key.bias",
    # "bert.encoder.layer.10.attention.self.value.weight",
    # "bert.encoder.layer.10.attention.self.value.bias",
    # "bert.encoder.layer.10.attention.output.dense.weight",
    # "bert.encoder.layer.10.attention.output.dense.bias",
    # "bert.encoder.layer.10.attention.output.LayerNorm.weight",
    # "bert.encoder.layer.10.attention.output.LayerNorm.bias",
    # "bert.encoder.layer.10.intermediate.dense.weight",
    # "bert.encoder.layer.10.intermediate.dense.bias",
    # "bert.encoder.layer.10.output.dense.weight",
    # "bert.encoder.layer.10.output.dense.bias",
    # "bert.encoder.layer.10.output.LayerNorm.weight",
    # "bert.encoder.layer.10.output.LayerNorm.bias",
    "bert.encoder.layer.11.attention.self.query.weight",
    "bert.encoder.layer.11.attention.self.query.bias",
    "bert.encoder.layer.11.attention.self.key.weight",
    "bert.encoder.layer.11.attention.self.key.bias",
    "bert.encoder.layer.11.attention.self.value.weight",
    "bert.encoder.layer.11.attention.self.value.bias",
    "bert.encoder.layer.11.attention.output.dense.weight",
    "bert.encoder.layer.11.attention.output.dense.bias",
    "bert.encoder.layer.11.attention.output.LayerNorm.weight",
    "bert.encoder.layer.11.attention.output.LayerNorm.bias",
    "bert.encoder.layer.11.intermediate.dense.weight",
    "bert.encoder.layer.11.intermediate.dense.bias",
    "bert.encoder.layer.11.output.dense.weight",
    "bert.encoder.layer.11.output.dense.bias",
    "bert.encoder.layer.11.output.LayerNorm.weight",
    "bert.encoder.layer.11.output.LayerNorm.bias",
    "bert.pooler.dense.weight",
    "bert.pooler.dense.bias",
    "cls.seq_relationship.weight",
    "cls.seq_relationship.bias",
]
for name, param in model.named_parameters():
    if name not in weight_name_list:
        param.requires_grad = False

qa_dataset = SADataset(tokenizer, conf)
train_dataset_len = int(0.8 * qa_dataset.data_len)
val_dataset_len = qa_dataset.data_len - train_dataset_len
train_dataset, val_dataset = random_split(qa_dataset, lengths=[train_dataset_len, val_dataset_len],
                                          generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))

model.train()
train_acc = 0
val_acc = 0
max_val_acc = 0.72
for epoch in range(num_epochs):
    metric_train = load_metric("./util/accuracy.py")

    for batch_index, train_batch in enumerate(train_dataloader):
        train_batch = {k: v.to(device) for k, v in train_batch.items()}
        outputs = model(**train_batch)
        pred_train = torch.argmax(outputs.logits, dim=-1)
        metric_train.add_batch(predictions=pred_train, references=train_batch["labels"])
        train_acc = metric_train.compute()["accuracy"]

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        loss_data = loss.data.item()

        if batch_index % 100 == 0:
            metric_val = load_metric("./util/accuracy.py")
            model.eval()
            for val_batch in val_dataloader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                with torch.no_grad():
                    outputs = model(**val_batch)

                pred_val = torch.argmax(outputs.logits, dim=-1)
                metric_val.add_batch(predictions=pred_val, references=val_batch["labels"])
            val_acc = metric_val.compute()["accuracy"]

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                torch.save(model.state_dict(), conf["train"]["save_path"])
                print(f">> save model success!")

        progress_bar.set_postfix(loss=loss_data, train_acc=train_acc, val_acc=val_acc)
