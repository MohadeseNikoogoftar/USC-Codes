import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import random



edges = pd.read_csv("elliptic_txs_edgelist.csv")
classes = pd.read_csv("elliptic_txs_classes.csv")
classes = classes[classes['class'] != 'unknown']

print("Total labeled transactions:", len(classes))




G_full = nx.DiGraph()

for _, row in edges.iterrows():
    G_full.add_edge(row['txId1'], row['txId2'])

print("Full graph nodes:", G_full.number_of_nodes())
print("Full graph edges:", G_full.number_of_edges())



START_NODE = random.choice(list(classes['txId']))
print("Starting BFS from:", START_NODE)

visited = set([START_NODE])
queue = [START_NODE]
MAX_NODES = 2000

while queue and len(visited) < MAX_NODES:
    current = queue.pop(0)
    neighbors = list(G_full.successors(current)) + list(G_full.predecessors(current))
    
    for n in neighbors:
        if n not in visited:
            visited.add(n)
            queue.append(n)
        if len(visited) >= MAX_NODES:
            break

G = G_full.subgraph(visited).copy()

print("Subgraph nodes:", G.number_of_nodes())
print("Subgraph edges:", G.number_of_edges())



print("Computing structural features...")

baseline_features = {}
motif16 = {}
two_hop_mean = {}

for node in tqdm(G.nodes()):
    
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    preds = list(G.predecessors(node))
    
    baseline_features[node] = (in_deg, out_deg)
    
    if len(preds) >= 2:
        motif16[node] = len(preds)
    else:
        motif16[node] = 0
    
    if len(preds) > 0:
        two_hop_mean[node] = np.mean([G.in_degree(p) for p in preds])
    else:
        two_hop_mean[node] = 0



label_dict = dict(zip(classes['txId'], classes['class']))

X_baseline = []
X_full = []
y = []

for node in G.nodes():
    if node not in label_dict:
        continue

    in_deg, out_deg = baseline_features[node]
    m16 = motif16[node]
    m16_norm = m16 / (in_deg + 1)
    hop2 = two_hop_mean[node]

    X_baseline.append([in_deg, out_deg])
    X_full.append([in_deg, out_deg, m16, m16_norm, hop2])

    y.append(1 if label_dict[node] == "1" else 0)

X_baseline = np.array(X_baseline)
X_full = np.array(X_full)
y = np.array(y)

print("Final dataset size:", len(y))



Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.3, random_state=42, stratify=y
)

Xf_train, Xf_test, _, _ = train_test_split(
    X_full, y, test_size=0.3, random_state=42, stratify=y
)



scale_pos_weight = (len(y) - sum(y)) / sum(y)


print("\nTraining Baseline Model...")
model_base = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
model_base.fit(Xb_train, y_train)

pred_base = model_base.predict_proba(Xb_test)[:,1]
auc_base = roc_auc_score(y_test, pred_base)



print("Training Structural Model (fan-in + 2-hop)...")
model_full = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
model_full.fit(Xf_train, y_train)

pred_full = model_full.predict_proba(Xf_test)[:,1]
auc_full = roc_auc_score(y_test, pred_full)


print("\n============================")
print("Baseline AUC:", round(auc_base, 4))
print("Structural Model AUC:", round(auc_full, 4))
print("============================")

if auc_full > auc_base:
    print("Structural features improved performance ")
else:
    print("No significant improvement")