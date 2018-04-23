import csv
from nltk.tokenize import word_tokenize

lines = open('train.csv', 'rb').readlines()
lines = lines[1:] #removing the header

tokens = []
train_data = []
train_labels = []
for line in lines:
    train_data.append(line.split("\t")[1])
    tempLabel = line.split("\t")[2]
    if float(tempLabel)>0.5:
        tempLabel = 1.0;
    else:
        tempLabel = 0.0;
    train_labels.append(tempLabel)
    # tokens.append(word_tokenize(line))
    # print(train_data)
    # print(train_labels)





