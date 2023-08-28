import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from brain import Net
from Neurons import token, bag_of_words, stem
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()

with open('feelings.json', 'r') as f:
    feelings = json.load(f)


word_list = []
tags = []
xy = []

for feel in feelings['feelings']:
    tag = feel['tag']
    tags.append(tag)

    for pattern in feel['patterns']:
        print(pattern)
        w = token(pattern)
        word_list.extend(w)

        xy.append((w, tag))

ignore_words = [',', '?', '/', '.', '!']
word_list = [stem(w) for w in word_list]
word_list = [w for w in word_list if w not in ignore_words]
word_list = sorted(set(word_list))
tags = sorted(set(tags))


x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, word_list)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.002
input_size = len(x_train[0])
print(input_size)
hidden_size = 8
output_size = len(tags)
print(output_size)

print("Training the model...")


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()

train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_size, output_size, hidden_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device=device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item() :.4f}')

print(f"Final loss: {loss.item(): .4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "word_list": word_list,
    "tags": tags
}

FILE = "TrainData.pth"
torch.save(data, FILE)

print(f"Training complete , File saved to {FILE}")
