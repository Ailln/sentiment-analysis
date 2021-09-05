# Sentiment Analysis

😄😐😠 情感分析（使用 emoji 可视化）

## 1 序言

众所周知，机器学习任务大都可以被分为「分类」和「回归」两种任务，它们之间的区分是看数据时离散还是连续的。因此我们的情感分析任务可以被看作成一个「多分类任务」。

为了更好地展示情感的变化，我使用 `emoji` 表情对分类结果进行了可视化，虽然不能完全展示出原有标签的意思，但是大致可以区分出来，参见 [DEMO](https://www.dovolopor.com/sentiment-analysis) 。

![Sentiment Analysis](./data/sentiment-analysis-v0.1.0.png)

## 2 数据

本实验的数据来源于 `NLPCC2014` 的微博情感分析测评任务，共 `48876` 条样本。这些数据中包含 `8` 个不同的类别，分别为：

- none: 😐
- happiness: 🥰
- like: 😍
- surprise: 😱
- disgust: 😞
- anger: 😠️
- sadness: 😥
- fear: 😨

请点击[这里](https://github.com/DinghaoXi/chinese-sentiment-datasets)下载数据，然后把 `Nlpcc2014Train.tsv` 文件放入 `./data/` 路径下面。

## 3 快速上手

本实验在下面环境中开发（尽可能保持一致）：

- Ubuntu 18.04 LTS+
- Python 3.6 +
- Anaconda 3

```bash
# 安装
conda create -n "sa" python==3.7.9
conda activate sa

cd sentiment-analysis
pip install -r requirements.txt

# 训练
python -m app.train

# 测试
python -m app.test

# 提供 API 服务
python -m server.app
# 测试 API
curl http://127.0.0.1:8012/sentimentAnalysis?text=%E6%88%91%E5%BE%88%E5%BC%80%E5%BF%83
```


## 4 常见问题

### 4.1 预训练模型 bert-base-chinese 无法下载

手动下载 bert-base-chinese 预训练模型
```bash
# 参考 https://huggingface.co/bert-base-chinese
sudo apt install git-lfs
git lfs install

cd save
git clone https://huggingface.co/bert-base-chinese
# 下载只需要把 config/default.yaml 中的 train: pre_train_model: 的值
# 由 bert-base-chinese 改为 ./save/bert-base-chinese 
```
### 4.2 无法使用 GPU

在 [对照表](https://pytorch.org/get-started/previous-versions/) 中找到合适版本的 torch 进行安装
```bash
# 以 cuda10.1 为例
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### 4.3 效果不好？

试试调节超参。

> 我自己训练的结果也只能达到 `75%` 左右...

另外 train.py 中 weight_name_list 是权重列表，你可以根据实际情况决定哪些权重需要微调。

## 5 参考

- [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)
- [Ubuntu 系统镜像下载](https://cn.ubuntu.com/download)
- [Anaconda 个人版](https://www.anaconda.com/products/individual#)
- [TUNA 清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/)

## 6 许可证

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)
