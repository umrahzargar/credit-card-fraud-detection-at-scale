# Credit Card Fraud Detection at Scale using Distributed SMOTEBagging

## Project Overview

This project implements and compares distributed SMOTEBagging strategies in Apache Spark for credit card fraud detection under extreme class imbalance.

The dataset contains legitimate and fraudulent credit card transactions, where fraudulent transactions represent only around 0.17% of the data. The goal of this project was to evaluate how different distributed oversampling and bagging designs affect fraud detection performance, scalability, and communication overhead.

The project was implemented using PySpark on Databricks and compares three distributed SMOTEBagging variants.

---

## Problem Statement

Fraud detection is a highly imbalanced classification problem. In real-world transaction data, fraudulent cases are rare but extremely important. Standard machine learning models can become biased toward the majority class and fail to detect fraudulent transactions.

SMOTE and bagging are useful for imbalanced classification, but applying them in a distributed Spark environment is challenging because minority examples may be split unevenly across partitions. This project explores the trade-off between:

* reducing communication overhead
* improving minority-class representation
* maintaining fraud detection performance
* scaling across partitions and dataset sizes

---

## Dataset

This project uses the public Kaggle Credit Card Fraud Detection dataset.

Dataset summary:

* 284,807 total transactions
* 492 fraud transactions
* Fraud class approximately 0.17% of all records
* 30 feature columns including PCA-transformed features
* Binary target: fraud vs non-fraud

The dataset was split into training and test sets using an 80/20 stratified split before scaling or oversampling to prevent data leakage.

---

## Tools and Technologies

* Python
* PySpark
* Apache Spark
* Azure Databricks / Databricks
* NumPy
* Pandas
* Scikit-learn
* Decision Trees
* SMOTE
* Bagging
* Matplotlib / Seaborn

---

## Methodology

### Shared Preprocessing

The preprocessing pipeline included:

* dropping the `Time` column
* stratified 80/20 train-test split
* standardising the `Amount` feature using a scaler fitted only on training data
* keeping PCA-transformed features unchanged
* assembling features into a Spark-compatible feature representation

All variants used the same preprocessing, base learner, and evaluation setup so that performance differences could be attributed mainly to distributed design choices.

---

## Distributed SMOTEBagging Variants

### Variant 1: Local Partition-Level SMOTEBagging

Variant 1 performs random shuffling, oversampling, bagging, and model training independently within each Spark partition.

This design minimises communication overhead, but it is sensitive to partition composition. If some partitions contain too few fraud examples, local SMOTE becomes unreliable.



---

### Variant 2: Stratified-Repartitioned Local SMOTEBagging

Variant 2 introduces stratified repartitioning before local SMOTEBagging. This ensures each partition receives a more reliable share of fraud and non-fraud examples.

Two strategies were explored:

* V2A: more uniform distribution using stronger global sorting
* V2B: lighter single-shuffle strategy with sufficient minority coverage

V2B provided the best balance between runtime and reliable fraud detection.

---

### Variant 3: Global Minority Broadcast SMOTEBagging

Variant 3 broadcasts the full minority fraud training set to each worker while keeping the majority class distributed.

This improves global minority-neighbour access for SMOTE, producing strong ranking performance, but it may require threshold tuning because default decision thresholds can produce more false positives.


---

## Evaluation Metrics

The models were evaluated using:

* Precision
* Recall
* F1-score
* G-Mean
* AUC-ROC
* AUPRC
* False positives
* Training time
* Partition scaling behaviour
* Dataset size scaling behaviour

AUPRC was treated as especially important because the dataset is extremely imbalanced.

---

## Results

At the standard threshold of 0.5, the variants showed different trade-offs:

| Metric        |     V1 |    V2A |    V2B |     V3 |
| ------------- | -----: | -----: | -----: | -----: |
| Precision     | 0.8737 | 0.8830 | 0.8723 | 0.2217 |
| Recall        | 0.7685 | 0.7685 | 0.7593 | 0.8889 |
| F1            | 0.8177 | 0.8218 | 0.8119 | 0.3549 |
| G-Mean        | 0.8766 | 0.8766 | 0.8713 | 0.9400 |
| AUC-ROC       | 0.9767 | 0.9570 | 0.9522 | 0.9846 |
| AUPRC         | 0.7509 | 0.8407 | 0.8311 | 0.7305 |
| Training Time | 51.57s | 53.47s | 20.61s | 29.10s |


---

## Scalability Findings

### Partition Sensitivity

All variants were tested across 2, 4, 8, and 16 partitions.

Key finding:

* V1 achieved the strongest speedup but was most sensitive to minority distribution.
* V2B provided a balanced trade-off between runtime and stable performance.
* V3 was the most stable in AUC-ROC but had less efficient speedup due to broadcast overhead.
<img width="937" height="286" alt="image" src="https://github.com/user-attachments/assets/d89f5fe2-b92c-43e8-adb6-2e4b9ce573c0" />


---

### Dataset Size Scaling

The dataset was sampled at 25%, 50%, 75%, and 100% while preserving the class imbalance ratio.

Key finding:

* All variants showed sub-linear runtime growth as data size increased.
* AUC-ROC remained stable across data fractions.
* V2B and V3 produced stronger AUPRC than V1 due to better minority-neighbour coverage.

<img width="944" height="276" alt="image" src="https://github.com/user-attachments/assets/bb410d47-71f7-434e-8f52-6e424b91dc9b" />


---

## Key Takeaways

This project found that no single distributed SMOTEBagging design dominates across all dimensions.

Deployment guidance:

* **V1** is best when communication overhead must be minimised.
* **V2B** is best when reliable minority coverage and practical fraud detection performance are required.
* **V3** is best when ranking quality matters and the minority set is small enough to broadcast efficiently.

---

## Project Contributions

This project demonstrates:

* distributed imbalanced classification using Apache Spark
* custom SMOTE implementation using partition-level logic
* comparison of local, stratified, and global minority-broadcast designs
* fraud detection under extreme class imbalance
* runtime scaling across partitions and dataset sizes
* evaluation using threshold-dependent and threshold-independent metrics
* practical deployment guidance for distributed oversampling strategies

---

## Repository Structure

```text
credit-card-fraud-detection-at-scale/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── distributed_smote_bagging_fraud_detection.ipynb
│
├── reports/
│   └── credit_card_fraud_spark_report.pdf
│
├── data/
│   └── README.md
│
└── images/
    ├── variant_1_pipeline.png
    ├── variant_2_pipeline.png
    ├── variant_3_pipeline.png
    ├── classification_results.png
    ├── partition_sensitivity.png
    └── dataset_size_scaling.png
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/umrahzargar/credit-card-fraud-detection-at-scale.git
```

2. Navigate into the project folder:

```bash
cd credit-card-fraud-detection-at-scale
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Kaggle Credit Card Fraud Detection dataset and place `creditcard.csv` in the `data/` folder.

5. Open the notebook:

```bash
jupyter notebook notebooks/distributed_smote_bagging_fraud_detection.ipynb
```

For full distributed execution, run the notebook in a Spark or Databricks environment.

---

## Limitations and Future Work

The experiments were conducted on a limited dataset scale, where all variants completed within under a minute. Larger datasets may reveal stronger differences in Spark scalability.

Future improvements include:

* approximate nearest-neighbour methods such as HNSW or FAISS
* distributed majority voting using Spark reduce operations
* probability calibration
* stronger threshold optimisation
* full weak and strong scaling experiments across different cluster sizes
* deployment of a monitoring dashboard for fraud detection results

---

## Author

**Umrah**

* GitHub: [github.com/umrahzargar](https://github.com/umrahzargar)
* LinkedIn: [linkedin.com/in/umrah-zargar](https://www.linkedin.com/in/umrah-zargar)
