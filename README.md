# NLI 推理、幻觉检测与偏见评估实验

本项目完成了自然语言推理（NLI）、大语言模型幻觉检测（Hallucination Detection）以及语言模型偏见评估（Bias Evaluation）三个任务，基于 Huggingface Transformers 框架和公开数据集进行实验。

## 项目结构

nli-hallucination-bias-eval/

├── README.md

├── roberta_finetune_nli.py      # RoBERTa 微调脚本

├── ft_roberta_matched.csv       # 微调模型在 matched 上的预测结果

├── ft_roberta_mismatched.csv     # 微调模型在 mismatched 上的预测结果

├── roberta_hallucination.csv     # 幻觉检测预测结果

├── nli_zero_shot_eval.ipynb      # 零样本推理实验

├── hallucination_detection.ipynb   # 幻觉检测实验

├── bias_crows_pairs_eval.ipynb    # 偏见评估实验

## 主要实验内容

### 1️⃣ NLI 推理

- 零样本推理使用 `facebook/bart-large-mnli` 和 `roberta-large-mnli`
- 微调实验使用 `roberta-base` 模型，在 MultiNLI 上训练 3 个 epoch

### 2️⃣ 幻觉检测

- 使用 `wikibio-gpt3-hallucination` 数据集
- 以 NLI 的方式检测 GPT-3 生成文本中的事实幻觉

### 3️⃣ 偏见评估

- 使用 CrowS-Pairs nationality 子集
- 使用 `bert-base-uncased` 和 `roberta-base` 对比伪似然分数 (PLL)，评估模型是否偏向刻板印象句

## 实验指标

- **NLI 微调模型**在 matched / mismatched 上准确率达 76.4% / 77.2%
- **幻觉检测模型**精确率 79.5%，召回率 20.1%，F1 为 32.1%
- **偏见评估**中 RoBERTa 表现出 60.38% 的偏向率，BERT 为 56.6%

## 部署环境

本实验在Colab上进行，使用T4-GPU，无需手动部署

