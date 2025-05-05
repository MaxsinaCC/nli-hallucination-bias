# NLI, Hallucination Detection, and Bias Evaluation

This repository presents three experiments conducted using Huggingface Transformers on Natural Language Inference (NLI), hallucination detection from GPT-3, and bias measurement in pretrained language models.

## Project Structure

nli-hallucination-bias-eval/

├── README.md

├── roberta_finetune_nli.py      # Fine-tuning script

├── ft_roberta_matched.csv       # Predictions on matched set

├── ft_roberta_mismatched.csv     # Predictions on mismatched set

├── roberta_hallucination.csv     # Hallucination detection results

├── nli_zero_shot_eval.ipynb      # Zero-shot evaluation notebook

├── hallucination_detection.ipynb   # Hallucination detection notebook

├── bias_crows_pairs_eval.ipynb    # Bias evaluation notebook

## Experiment Summary

### 1️⃣ NLI Evaluation

- **Zero-shot:** Evaluated `facebook/bart-large-mnli` and `roberta-large-mnli` on MultiNLI.
- **Fine-tuning:** Trained `roberta-base` on 50k subset of MultiNLI for 3 epochs.

### 2️⃣ Hallucination Detection

- Dataset: `wikibio-gpt3-hallucination`
- Approach: If hypothesis contradicts the premise, label as hallucinated.

### 3️⃣ Bias Evaluation

- Dataset: CrowS-Pairs (`nationality` subset)
- Evaluated `bert-base-uncased` and `roberta-base` using pseudo-log-likelihood (PLL)

## Results Summary

- **Fine-tuned RoBERTa-NLI:** Accuracy = 76.4% / 77.2%
- **Hallucination Detection:** Precision = 79.5%, Recall = 20.1%, F1 = 32.1%
- **Bias Evaluation:** RoBERTa (60.38%), BERT (56.6%) showed stereotypical preferences

## Dependencies

This experiment was conducted on Colab, using T4-GPUs without manual deployment.
