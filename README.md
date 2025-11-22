# Multi-Label Classification of Biological Assay Activity Using Machine & Deep Learning
This project evaluated the performance of classical machine learning model versus cutting-edge deep learning architectures for predicting multi-label toxicity
across the 12 biological assays of the Tox21 dataset. Using 3,074 cleaned molecular samples,
logistic regression with PCA, random forest, XGBoost, a graph convolutional network (GCN),
and a graph transformer were benchmarked and evaluated using PR-AUC (Precision Recall -
Area Under Curve). The classical models provided fast, interpretable baselines but were limited
in capturing molecular structure, while GCNs and graph transformers leveraged the nodes and
edges formed through chemical bonds to compute their predictions. Results showed that logistic
regression performed the weakest, whereas the graph transformer achieved the highest average
PR-AUC despite the small dataset size. Random forest unexpectedly outperformed XGBoost,
likely due to XGBoost’s sensitivity to class imbalance for the cross-entropy loss optimization.
GCN performance was constrained by limited data and tuning resources. Although none of the
models reached high PR-AUC values, this study highlighted the advantages of state of the art
models and the limitations imposed by small imbalanced datasets. In the future, incorporating
large scale pretraining, more intensive hyperparameter optimization, and more expressive graph
transformer architectures could strengthen predictive performance.
