import tensorflow.contrib as learn
import numpy as np
import ReadData as read

print("Loading data...")
x_text, y = read.load_data()


max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

print (max_doocument_length)
