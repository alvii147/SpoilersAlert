import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

train = 0
test = 1

spoiler_data = []
with open("spoiler_data.txt", 'r') as datafile:
    lines = datafile.readlines()
    for line in lines:
        spoiler_data.append(line.rstrip())

test_sentences = [
    "Jamie manages to kill Euron",
    "The Mountain is scary",
    "Cersei is arguably one of the most hated characters",
    "Jon kills Dany because he knows there's no other way",
    "Today is a beautiful day",
    "Who knew the next leader of Westeros would be Bran",
    "Bran is the three eyed raven",
    "I think Jaime is a better person in this season"
]

if train == 1:
    train_data = []
    train_labels = []
    with open("train_data.txt", 'r') as datafile:
        lines = datafile.readlines()
    for line in lines:
        splitline = line.split()
        strdata = ""
        for i in range(len(splitline) - 1):
            strdata = strdata + splitline[i] + " "
        train_data.append(strdata.rstrip())
        train_labels.append(int(splitline[len(splitline) - 1]))
    
    embed = hub.Module("UniversalSentenceEncoder/")
    sentences = train_data + spoiler_data
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sentence_embeddings = session.run(embed(sentences))

    inner_prod = np.inner(sentence_embeddings, sentence_embeddings)
    similarity = []
    for i in range(len(train_data)):
        temp = []
        for j in range(len(train_data), len(inner_prod[i])):
            temp.append(inner_prod[i][j])
        temp2 = []
        for k in range(0, len(temp), 3):
            temp2.append((temp[k] + temp[k + 1] + temp[k + 2])/3)
        similarity.append(temp2)
    similarity = np.asarray(similarity)
    x = np.asarray([max(i) for i in similarity])
    y = np.asarray(train_labels)

    def sigmoid(X, a, b):
        return 1/(1+np.exp(a*X+b))
    
    param, param_cov = curve_fit(sigmoid, x, y)
    
    with open("param.txt", "w+") as datafile:
        datafile.write(str(param[0]) + "\n")
        datafile.write(str(param[1]) + "\n")

    y_pred = [sigmoid(i, param[0], param[1]) for i in x]
    plt.plot(x, y, 'o', color = "yellow")
    plt.plot(x, y_pred, 'o', color = "purple")
    plt.show()

if test == 1:
    with open("param.txt", 'r') as datafile:
        lines = datafile.readlines()
        a = float(lines[0].rstrip())
        b = float(lines[1].rstrip())
    
    embed = hub.Module("UniversalSentenceEncoder/")
    sentences = test_sentences + spoiler_data
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sentence_embeddings = session.run(embed(sentences))
    
    inner_prod = np.inner(sentence_embeddings, sentence_embeddings)
    similarity = []
    for i in range(len(test_sentences)):
        temp = []
        for j in range(len(test_sentences), len(inner_prod[i])):
            temp.append(inner_prod[i][j])
        temp2 = []
        for k in range(0, len(temp), 3):
            temp2.append((temp[k] + temp[k + 1] + temp[k + 2])/3)
        similarity.append(temp2)
    similarity = [max(i) for i in similarity]
    
    for i in range(len(test_sentences)):
        print("Statement: " + test_sentences[i])
        print("Spoiler Probability: " + str(round(similarity[i] * 100, 2)) + '%')