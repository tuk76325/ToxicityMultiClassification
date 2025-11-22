import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Data Loading and Preprocessing
df = pd.read_csv('tox21.csv')
print(df.shape)
print(df.dtypes)

# Drop values
df.drop(columns=['mol_id'], inplace=True)
df = df.dropna()

# New Shape of df
print(df.shape)

assayCols = df.columns[:12]
molCols = df.columns[-1]
# Positive Labels per Assay
pos_rate = df[assayCols].mean() * 100
ax = pos_rate.plot(kind='bar', figsize=(12,5))
plt.title("Positive Label % per Assay")
plt.ylabel("Percent Activity")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(pos_rate): # add labels to plot
    ax.text(i, v, f"{v:.2f}%", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# Heatmap correlation
def jaccard_similarity_matrix(df, cols):
    n = len(cols)
    jaccard = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = df[cols[i]].astype(bool)
            b = df[cols[j]].astype(bool)
            intersection = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            jaccard[i, j] = intersection / union if union != 0 else 0
    return pd.DataFrame(jaccard, index=cols, columns=cols)

jaccard_df = jaccard_similarity_matrix(df, assayCols)
sns.heatmap(jaccard_df, cmap="plasma", annot=True, fmt=".2f")
plt.title("Jaccard Similarity Between Assay Labels (Shared Positives)")
plt.tight_layout()
plt.show()

# Get physical and chemical properties of each molecule
df["mol"] = df[molCols].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()]  # Drop unmappable molecules
print(df.columns)
print(df.shape)
print(df.dtypes)



def compute_features(mol):
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumAtoms": mol.GetNumAtoms()
    })

features = df["mol"].apply(compute_features)
df = pd.concat([df, features], axis=1)
print(df.columns)
print(df)

    #Molwt = molecular weight
    #LogP = chemical's lipophilicity (fat-loving) or hydrophobicity (water-fearing), high logP means a compound is more lipophilic, low or negative logP means it is more hydrophilic
    #HBD = Num of hydrogen atoms connected to nitrogen https://www.researchgate.net/figure/Hydrogen-bond-donor-HBD-and-hydrogen-bond-acceptor-HBA-sites-for-trizaole-Tz-left_fig5_346474495
    #HBA = Hydrogen bond acceptors which are nitrogen and oxygen atoms
    #TPSA = Topological Polar Surface Area

# distributions and box plots
df[["MolWt","LogP","NumAtoms"]].hist(bins=30, figsize=(12,4))
plt.suptitle("Descriptor Distributions")
plt.tight_layout()
plt.show()

descriptor_cols = ["LogP", "HBD", "HBA", "NumAtoms"]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[descriptor_cols])
plt.title("Distribution of Chemical Descriptors")
plt.tight_layout()
plt.show()

sns.boxplot(data=df[["MolWt", "TPSA"]])
plt.title("Distribution of Molecular Weight")
plt.show()

# Heatmap correlation
allDescriptor_cols = ["MolWt", "LogP", "HBD", "HBA", "TPSA", "NumAtoms"]
corr = df[allDescriptor_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Label Correlation Between Descriptors")
plt.show()

linearDF = df.copy(deep=True)
linearDF.drop(columns=["NumAtoms"], inplace=True)

# Train Test Split and Scaling for linear models (removing numAtoms b/c correlation with molWeight)
X = linearDF[linearDF.columns[14:]]
Y = linearDF[linearDF.columns[0:12]]

print(X.columns)
print(Y.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=7406
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Original shape: {X.shape}, PCA shape: {X_pca.shape}")

print("Logistic Regression with PCA: \n")

# Logistic Regression
clf = MultiOutputClassifier(LogisticRegression()).fit(X_pca, y_train)
results = clf.predict(X_test_pca)

acc = accuracy_score(y_test, results)
print(acc)

#ROC-AUC, not good because it does not handle class imbalance well and is inflated by 0s
y_proba = np.column_stack([
    est.predict_proba(X_test_pca)[:, 1]
    for est in clf.estimators_
])

plt.figure(figsize=(18, 12))

for i, assay in enumerate(Y.columns):
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_proba[:, i])
    auc_score = auc(fpr, tpr)

    plt.subplot(3, 4, i+1)
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"ROC Curve: {assay}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

plt.tight_layout(h_pad=3)
plt.show()

#PR-AUC, captures how well the model finds and confirms the true positives, across all possible thresholds
plt.figure(figsize=(20, 16))

for i, assay in enumerate(Y.columns):
    precision, recall, _ = precision_recall_curve(y_test.iloc[:, i], y_proba[:, i])
    ap_score = average_precision_score(y_test.iloc[:, i], y_proba[:, i])

    plt.subplot(3, 4, i + 1)
    plt.step(recall, precision, where='post')
    plt.title(f"PR Curve: {assay} (AP={ap_score:.3f})", fontsize=10, y=1.03)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])

plt.tight_layout(h_pad=3)
plt.show()

pr_auc_scores_lg = []
for i, col in enumerate(y_train.columns):
    pr  = average_precision_score(y_test[col], y_proba[:, i])
    pr_auc_scores_lg.append(pr)
    print(f"{col:10s} | PR-AUC: {pr:.3f}")

print(f"Mean PR-AUC: {np.mean(pr_auc_scores_lg):.3f}\n")

#If 5% of samples are positive → random PR-AUC baseline = 0.05
# So if logistic regression PR-AUC = 0.08:
# The model is barely outperforming random guessing on toxic detection.

# # Random Forest
# rf = RandomForestClassifier(random_state=7406, n_jobs=-1)
#
# # Wrap in multi-label container
# multi_rf = MultiOutputClassifier(rf)
#
# param_grid = {
#     'estimator__n_estimators': [50, 100, 200, 250, 300, 350],
#     'estimator__max_depth': [None, 10, 20],
# }
#
# # Grid search
# grid_search = GridSearchCV(
#     estimator=multi_rf,
#     param_grid=param_grid,
#     cv=KFold(n_splits=6, shuffle=True, random_state=7406),
#     scoring='average_precision',
#     verbose=2,
#     n_jobs=-1
# )
#
# grid_search.fit(X_train, y_train)
#
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)

# Best parameters: {'estimator__max_depth': 10, 'estimator__n_estimators': 350}
# Best score: 0.13837645234518375

print("Random Forest: \n")

rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=10,
    random_state=7406,
    class_weight="balanced_subsample",
    n_jobs=-1
)

rf_clf = MultiOutputClassifier(rf, n_jobs=-1)
rf_clf.fit(X_train_scaled, y_train)

# Predict probabilities and get PR-AUC curve
y_proba_rf = np.stack([est.predict_proba(X_test_scaled)[:, 1] for est in rf_clf.estimators_], axis=1)

pr_auc_scores = []

for i, col in enumerate(y_train.columns):
    pr  = average_precision_score(y_test[col], y_proba_rf[:, i])
    pr_auc_scores.append(pr)
    print(f"{col:10s} | PR-AUC: {pr:.3f}")

print(f"Mean PR-AUC: {np.mean(pr_auc_scores):.3f}")

for i, assay in enumerate(Y.columns):
    precision, recall, _ = precision_recall_curve(y_test.iloc[:, i], y_proba_rf[:, i])
    ap_score = average_precision_score(y_test.iloc[:, i], y_proba_rf[:, i])

    plt.subplot(3, 4, i + 1)
    plt.step(recall, precision, where='post')
    plt.title(f"rf PR Curve: {assay} (AP={ap_score:.3f})", fontsize=10, y=1.03)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])

plt.tight_layout(h_pad=3)
plt.show()

print("XGBoost: \n")

# xgb_base = XGBClassifier(
#     eval_metric='aucpr',
#     random_state=7406,
#     n_jobs=-1
# )
#
# # Wrap in MultiOutputClassifier
# multi_xgb = MultiOutputClassifier(xgb_base, n_jobs=-1)
#
# # Define hyperparameter grid
# param_grid = {
#     'estimator__n_estimators': [50, 100, 200, 250, 300, 350],
#     'estimator__max_depth': [3, 5, 10, 20],
#     'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2]
# }
#
# # GridSearchCV setup
# grid_search_xgb = GridSearchCV(
#     estimator=multi_xgb,
#     param_grid=param_grid,
#     cv=KFold(n_splits=6, shuffle=True, random_state=7406),
#     scoring='average_precision',
#     verbose=2,
#     n_jobs=-1
# )
#
# # Fit grid search
# grid_search_xgb.fit(X_train_scaled, y_train)
#
# # Print best parameters and score
# print("Best parameters:", grid_search_xgb.best_params_)  {'estimator__learning_rate': 0.01, 'estimator__max_depth': 20, 'estimator__n_estimators': 250}
# print("Best score (mean PR-AUC):", grid_search_xgb.best_score_)
#
# best_params = grid_search_xgb.best_params_

xgb_final = XGBClassifier(
    n_estimators=250,
    max_depth=20,
    learning_rate=0.01,
    eval_metric='aucpr',
    random_state=7406,
    n_jobs=-1
)

xgb_clf_final = MultiOutputClassifier(xgb_final, n_jobs=-1)
xgb_clf_final.fit(X_train_scaled, y_train)

# Predict probabilities and get PR-AUC curve
y_proba_xgb = np.stack([est.predict_proba(X_test_scaled)[:, 1] for est in xgb_clf_final.estimators_], axis=1)

pr_auc_scores_xgb = []

for i, col in enumerate(y_train.columns):
    pr = average_precision_score(y_test[col], y_proba_xgb[:, i])
    pr_auc_scores_xgb.append(pr)
    print(f"{col:10s} | PR-AUC: {pr:.3f}")

print(f"Mean PR-AUC: {np.mean(pr_auc_scores_xgb):.3f}\n")

plt.figure(figsize=(20, 16))
for i, assay in enumerate(Y.columns):
    precision, recall, _ = precision_recall_curve(y_test.iloc[:, i], y_proba_xgb[:, i])
    ap_score = average_precision_score(y_test.iloc[:, i], y_proba_xgb[:, i])
    plt.subplot(3, 4, i + 1)
    plt.step(recall, precision, where='post')
    plt.title(f"xgb PR Curve: {assay} (AP={ap_score:.3f})", fontsize=10, y=1.03)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])

plt.tight_layout(h_pad=3)
plt.show()

#GCN Model

# Build the GCN dataset
def mol_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atom type, degree, formal charge, etc.)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic())
        ])

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge Index
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])   # undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def safe_tensor(y):
    y = np.array(y, dtype=np.float32)
    return torch.tensor(y, dtype=torch.float32)


# Custom GCN
class GCNGlobal(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=12, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)  # graph-level embedding
        return self.lin(x)  # logits per graph

# Build PyG graphs
graph_list = []
df[assayCols] = df[assayCols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=assayCols)

for _, row in df.iterrows():
    g = mol_to_pyg(row["smiles"])
    if g is None:
        continue

    label_array = row[assayCols].astype(float).values
    # 2D shape (1, num_tasks)
    g.y = torch.tensor(label_array, dtype=torch.float).unsqueeze(0)
    graph_list.append(g)

print("Total graphs:", len(graph_list))

# Train/Test split
train_idx, test_idx = train_test_split(np.arange(len(graph_list)), test_size=0.2, random_state=7406)
train_graphs = [graph_list[i] for i in train_idx]
test_graphs = [graph_list[i] for i in test_idx]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32)


# Model
in_dim = graph_list[0].num_node_features
num_tasks = len(assayCols)
model = GCNGlobal(in_dim=in_dim, hidden_dim=128, out_dim=num_tasks)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Training
def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_graphs)

for epoch in range(50):
    loss = train_epoch()
    print(f"Epoch {epoch+1:02d} | Loss = {loss:.4f}")


# Evaluate
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.sigmoid(logits)
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(probs.cpu().numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)


# PR-AUC Curves
plt.figure(figsize=(20, 16))
gcn_pr_auc_scores = []

for i, assay in enumerate(assayCols):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    ap_score = average_precision_score(y_true[:, i], y_pred[:, i])
    gcn_pr_auc_scores.append(ap_score)

    plt.subplot(3, 4, i+1)
    plt.step(recall, precision, where='post')
    plt.title(f"GCN PR Curve: {assay} (AP={ap_score:.3f})", fontsize=10, y=1.03)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])

plt.tight_layout(h_pad=3)
plt.show()

print("GCN PR-AUC per assay:")
for assay, pr in zip(assayCols, gcn_pr_auc_scores):
    print(f"{assay:10s} | PR-AUC: {pr:.3f}")
print(f"\nMean GCN PR-AUC: {np.mean(gcn_pr_auc_scores):.3f}")

# Graph Transformer
train_idx, test_idx = train_test_split(np.arange(len(graph_list)), test_size=0.2, random_state=7406)
train_graphs = [graph_list[i] for i in train_idx]
test_graphs = [graph_list[i] for i in test_idx]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32)

# Define TransformerConv-based model
class TransformerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=12, heads=4, dropout=0.2):
        super().__init__()
        # Graph attention via TransformerConv
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout)
        self.lin = nn.Linear(hidden_dim*heads, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)  # graph-level embedding
        return self.lin(x)  # logits per graph

# Initialize model
in_dim = graph_list[0].num_node_features
num_tasks = len(assayCols)
model_tf = TransformerGCN(in_dim=in_dim, hidden_dim=128, out_dim=num_tasks)
optimizer = torch.optim.Adam(model_tf.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# ------------------------------
# Training loop
# ------------------------------
for epoch in range(50):
    model_tf.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model_tf(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    print(f"Epoch {epoch+1:02d} | Loss = {total_loss / len(train_graphs):.4f}")

# ------------------------------
# Evaluation & PR-AUC
# ------------------------------
model_tf.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        logits = model_tf(batch.x, batch.edge_index, batch.batch)
        probs = torch.sigmoid(logits)
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(probs.cpu().numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# PR-AUC Curves
plt.figure(figsize=(20, 16))
tf_pr_auc_scores = []

for i, assay in enumerate(assayCols):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    ap_score = average_precision_score(y_true[:, i], y_pred[:, i])
    tf_pr_auc_scores.append(ap_score)

    plt.subplot(3, 4, i+1)
    plt.step(recall, precision, where='post')
    plt.title(f"TransformerConv PR Curve: {assay} (AP={ap_score:.3f})", fontsize=10, y=1.03)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])

plt.tight_layout(h_pad=3)
plt.show()

# Print PR-AUC summary
print("TransformerConv PR-AUC per assay:")
for assay, pr in zip(assayCols, tf_pr_auc_scores):
    print(f"{assay:10s} | PR-AUC: {pr:.3f}")
print(f"\nMean TransformerConv PR-AUC: {np.mean(tf_pr_auc_scores):.3f}")