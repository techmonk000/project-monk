from listen import listenvoice
from voice import speak
import random
import json
import torch
from brain import Net
from Neurons import bag_of_words, token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("feelings.json") as json_data:
    feelings = json.load(json_data)

FILE = "traindata.pth"
data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
word_list = data["word_list"]
tags = data["tags"]
model_state = data["model_state"]

model = Net(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# creating the main model
Name = "Monk"


def Main():
    sentence = listenvoice()

    if sentence == "exit":
        exit()

    sentence = token(sentence)
    X = bag_of_words(sentence, word_list)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    prob = torch.softmax(output, dim=1)
    probs = prob[0][predicted.item()]

    if prob.item() > 0.75:
        for feel in feelings['feelings']:
            if tag == feel["tag"]:
                reply = random.choice(feel["responses"])
                speak(reply)

Main()

