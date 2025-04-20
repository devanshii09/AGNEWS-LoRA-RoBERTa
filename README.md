# Parameter-Efficient Fine-Tuning of RoBERTa for AGNEWS Classification using LoRA

**Authors:**  
Devanshi Bhavsar, Nikhil Arora  
New York University  
Contact: dnb7638@nyu.edu, na4063@nyu.edu

**Repository:**  
[AGNEWS-LoRA-RoBERTa GitHub](https://github.com/devanshii09/AGNEWS-LoRA-RoBERTa)

---

## Project Overview

This project presents a parameter-efficient approach to fine-tuning RoBERTa for the AGNEWS text classification task using **Low-Rank Adaptation (LoRA)**. We constrain the number of trainable parameters to under **1 million**, demonstrating that lightweight fine-tuning can still achieve **>95% validation accuracy**.

Instead of full fine-tuning, we inject LoRA adapters into RoBERTa’s attention modules while freezing the rest of the model. This technique drastically reduces training cost and makes deployment on resource-limited devices feasible.

---

## Directory Structure

```
.
├── main.py               # End-to-end fine-tuning and evaluation
├── model.py              # LoRA adapter implementation
├── utils.py              # Data preprocessing and metric utilities
├── test_unlabelled.pkl   # Unlabeled test set from Kaggle
├── submission.csv        # Final submission predictions
├── loss_curve.png        # Training loss plot
├── requirements.txt      # Package dependencies
└── README.md             # Project documentation
```

---

## Results

| Epoch | Train Loss | Validation Accuracy |
|-------|------------|---------------------|
| 1     | 0.2892     | 94.26%              |
| 2     | 0.1730     | 94.65%              |
| 3     | 0.1456     | **95.03%**          |

The final model uses **888K trainable parameters**, satisfying the <1M constraint.

---

## Methodology

- **Backbone:** `roberta-base` from HuggingFace
- **Tokenizer:** `RobertaTokenizerFast`, max sequence length = 128
- **LoRA Injection:** Applied to the Query and Value projections of all 12 transformer layers
- **Frozen Layers:** All RoBERTa backbone parameters (except LayerNorm and classifier)
- **LoRA Config:**  
  - Rank (`r`): 8  
  - Scaling factor (`α`): 24

---

## Training Configuration

- **Loss:** CrossEntropyLoss with label smoothing = 0.1
- **Optimizer:** AdamW with learning rate = 4e-4
- **Scheduler:** Linear with 10% warmup
- **Epochs:** 3  
- **Batch Size:** 32  
- **Validation Split:** 10% from the AGNEWS training set

---

## Reproducibility

All code and scripts necessary to reproduce the results are provided.  

### Requirements:
- Python ≥ 3.10  
- PyTorch 2.0  
- transformers==4.38  
- scikit-learn, pandas, datasets, matplotlib

### Steps to Reproduce:

1. Clone the repository  
2. Install dependencies  
   ```
   pip install -r requirements.txt
   ```
3. Run training:  
   ```
   python main.py
   ```
4. After training, generate predictions:  
   ```
   python main.py --predict
   ```

---

## Dataset

We use the [AGNEWS dataset](https://huggingface.co/datasets/ag_news), which includes 120,000 training and 7,600 test samples across 4 categories:  
- World  
- Sports  
- Business  
- Science/Technology  

We split the training set into 90% training and 10% validation.

> **Citation:**
> ```
> @misc{zhang2015character,
>   title={Character-level Convolutional Networks for Text Classification},
>   author={Xiang Zhang and Junbo Zhao and Yann LeCun},
>   year={2015},
>   eprint={1509.01626},
>   archivePrefix={arXiv},
>   primaryClass={cs.CL}
> }
> ```

---

## Evaluation and Inference

- **Evaluation:** Validation accuracy measured after each epoch.
- **Test Predictions:** Generated on the provided Kaggle test set.
- **Output:** `submission.csv` containing `ID` and `Label` columns.

---

## Observations

- LoRA-enabled tuning yields performance competitive with full fine-tuning.
- With only 888K parameters, our model surpasses 95% validation accuracy.
- No ensembling or test-time tricks were used—highlighting robustness.

---

## Limitations and Future Work

- Only Query and Value projections were modified. Exploring full LoRA injection (including Key and Output) could improve results.
- Current rank setting is uniform (r=8). Future work could explore non-uniform LoRA across layers.
- Cross-dataset generalization was not evaluated.

---

## Acknowledgements

We thank **Professor Chinmay Hegde** and the teaching staff of **Deep Learning (ECE-GY 9253)** at NYU for their guidance and support throughout the project.

---

## License

This project is open-source under the MIT License.
