# Diabetes Prediction using GCN (Graph Convolutional Network)

Dataset: 515 subjects, 308 voice/clinical features, binary diabetes label.

## Approach
1. Built a k-NN similarity graph over subjects (k=8, cosine similarity)
2. Trained a 2-layer GCN (built from scratch in PyTorch) for node classification
3. Achieved 94.17% test accuracy on unseen subjects

## Files
- `Untitled0.ipynb` — full code and outputs
- `my_own_graph.png` — the subject similarity graph
- `training_curve.png` — GCN training/validation accuracy over time 
