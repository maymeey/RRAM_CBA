import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

transform = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



class ANNModel(nn.Module) :
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ANNModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 50
losses = []

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

plt.figure()
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Epoch-Loss Graph')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()


test_data = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), 8, 8)  
        for i in range(images.size(0)):
            label = labels[i].item()
            pixel_values = images[i].numpy().flatten()  
            test_data.append([label] + pixel_values.tolist())

# 레이블 이름 설정
label_columns = [f'Label_{i}' for i in range(len(test_data))]


test_df = pd.DataFrame(test_data).T
test_df.columns = label_columns  

output_file_path = 'transposed_flattened_test_data.csv'
test_df.to_csv(output_file_path, index=False)
print(f"Test set saved in transposed format to {output_file_path}.")

correct = 0
total = 0 

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Accuracy on the test set: {accuracy:.2f}%")


weights = model.fc3.weight.data.numpy()


G_max, G_min = 0.0021, 0.00000511
#G_max, G_min = 1, 0


W_positive = np.maximum(weights, 0)
W_negative = -np.minimum(weights, 0)


max_abs_weight = np.max(np.abs(weights))
W_norm_positive = W_positive / max_abs_weight
W_norm_negative = W_negative / max_abs_weight


G_positive = W_norm_positive * (G_max - G_min) + G_min
G_negative = W_norm_negative * (G_max - G_min) + G_min
labels = [f'Label_{i}' for i in range(G_positive.shape[0])]


G_positive_df = pd.DataFrame(G_positive.T)
G_positive_df.columns = labels 
G_positive_df.loc[-1] = labels  
G_positive_df.index = G_positive_df.index + 1 
G_positive_df = G_positive_df.sort_index()  

G_negative_df = pd.DataFrame(G_negative.T)
G_negative_df.columns = labels
G_negative_df.loc[-1] = labels
G_negative_df.index = G_negative_df.index + 1
G_negative_df = G_negative_df.sort_index()

G_positive_df.to_csv('conductance_positive_ann_v3.csv', index=False)
G_negative_df.to_csv('conductance_negative_ann_v3.csv', index=False)
print("Conductance matrices saved with labels in 'conductance_positive_ann_v3.csv' and 'conductance_negative_ann_v3.csv'.")


label_outputs = defaultdict(list)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        for i in range(images.size(0)):
            label = labels[i].item()
            output = outputs[i].numpy()
            label_outputs[label].append(output)

fig, axes = plt.subplots(5, 2, figsize=(10, 12))
fig.suptitle('Output by Label', fontsize=16)

for i in range(10):
    avg_output = np.mean(label_outputs[i], axis=0)  
    ax = axes[i // 2, i % 2]
    ax.plot(avg_output, marker='o')
    ax.set_title(f'Label {i}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Current (A)')

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('mnist_label_output_plot.png')
print("Plot saved as 'mnist_label_output_plot.png'")

avg_output_data = {}

for i in range(10):
    avg_output = np.mean(label_outputs[i], axis=0)  
    avg_output_data[f'Label {i}'] = avg_output 

avg_output_df = pd.DataFrame(avg_output_data)
avg_output_df.to_csv('testset_avg_output.csv', index_label='Index')
print("Averaged output data saved to 'testset_avg_output.csv'")