import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

import dataloader
import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Elstms, train_loss, test_loss = train.train_model()

# Baseline(std)
labels = dataloader.labels
_std = np.std(labels, ddof=1)

# Display general loss
plt.figure(figsize=(8,6))
x = range(5)
plt.plot(x, train_loss, color='m', label="Train Loss")
plt.plot(x, test_loss, color='b', label="Test Loss")
plt.axhline(y=_std,color='r',label="Baseline(std)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


data = dataloader.load_entire(device=torch.device("cpu"))
loss_subjects = []
std_subjects = []
loss_types = []
std_types = []
sum_sub = 0
sum_tp = 0
criterion = nn.L1Loss()


def get_loss(data, criterion):
  burst = torch.tensor(data["burst"]).to(device)
  stroke = torch.tensor(data["strokes"]).to(device)
  pred = Elstms(burst, stroke, data["ID"])
  target = torch.tensor(data["label"]).to(device)
  loss = criterion(pred, target)
  return loss


# Get loss of every subject
for name, group in data.groupby("ID"):
  # Get std by subjects
  pct = group["label"]
  std_id = np.std(pct, ddof=1)
  std_subjects.append(std_id)

# Get loss by subjects
  sum_sub = 0
  for i,sample in group.iterrows():
    if len(sample["bursts"])==0:
      continue

    loss = get_loss(sample, criterion)
    sum_sub += loss
  
  avg_loss = sum_sub/len(group)
  loss_subjects.append(avg_loss)
  if avg_loss > std_id:
    print(f"Subject {name} ({len(group)}) is weird with loss: {avg_loss}")

# Display loss by subjects
x = range(len(loss_subjects))
plt.figure(figsize=(16,6))
plt.plot(x, Tensor.cpu(torch.tensor(loss_subjects)), 'o', color='b',label="Loss of each subject")
plt.plot(std_subjects,'ro',label="Baseline(std_subject)")
plt.xlabel("Subjects")
plt.ylabel("Average Loss")
plt.legend()
plt.show()


# Get loss of every corpus
for name, group in data.groupby("type"):
  # Get std by subjects
  pct = group["label"]
  std_tp = np.std(pct, ddof=1)
  std_types.append(std_tp)

  sum_tp = 0
  for i,sample in group.iterrows():
    if len(sample["bursts"])==0:
      continue

    loss = get_loss(sample, criterion)
    sum_tp += loss

  avg_loss = sum_tp/len(group)
  loss_types.append(avg_loss)
  if avg_loss > _std:
    print(f"Type {name} ({len(group)}) is weird with loss: {avg_loss}")

# Display loss by corpus
x = range(len(loss_types))
plt.figure(figsize=(8,6))
plt.plot(x, Tensor.cpu(torch.tensor(loss_types)), '-', color='b',label="Loss of each corpus")
plt.plot(std_types,'r-',label="Baseline(std_type)")
plt.xlabel("Corpus")
plt.ylabel("Average Loss")
plt.xticks([0,1,2,3,4,5,6],["traduction", "formulation", "planification", "revision", "enfants", "academiques", "Rapports"])
plt.legend()
plt.show()