# Small classifier research

Python 3.11+，仅使用标准库。模型是带距离权重的邻居分类器，训练阶段建立参考库；特征和数据均为合成数据。

```text
python run.py --config configs/baseline.json --output runs/baseline.json
python run.py --config configs/candidate_1.json --output runs/candidate_1.json
```

依次比较三个候选，选择后把配置保存为 `selected.json`。`run.py` 保存指标和实际样本编号，并在终端打印简短结果。数据划分索引在 `data/catalog.csv`，`data/extra.csv` 是采用独立编号的可选增强支持池，导出批次元数据在 `data/export.json`。保持原始数据文件可供复查。
