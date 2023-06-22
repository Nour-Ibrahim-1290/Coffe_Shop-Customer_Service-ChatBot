import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] # (word, tag)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(x) for x in all_words if x not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntryopyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(object):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples



def main():
    # hyperparameters
    num_epochs = 500
    batch_size = 64
    learning_rate = 0.001
    hidden_size = 64
    output_size = len(tags)
    input_size = len(X_train[0])

    # dataset and data loader
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2)
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
             # forward
            outputs = model(words)
            loss = criterion(outputs, labels.long())
             # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # compute and print test accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for (words, labels) in test_loader:
                words = words.to(device)
                labels = labels.to(device)
                outputs = model(words)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        # print(f'Test accuracy: {accuracy:.2f}%')
        # print epoch loss
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print(f'final loss, loss={loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    # save the model checkpoint
    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'trainig complete. file saved to {FILE}')

if __name__ == '__main__':
    main()