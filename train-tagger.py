import sys
import csv
import string
from collections import defaultdict

if __name__ == '__main__':
    # intialize core variables
    if len(sys.argv) == 5:
        trainingFile = str(sys.argv[1])
        transFile = str(sys.argv[2])
        emmisionsFile = str(sys.argv[3])
        laplaceFile = str(sys.argv[4])
    else:
        print("<Using Default Parameters>")
        trainingFile = 'train.txt'
        transFile = 'transitions.txt'
        emmisionsFile = 'emissions.txt'
        laplaceFile = 'laplace-tag-unigrams.txt'

    tag_bigrams = {}    # (POS,POS) = occurences
    word_unigrams = {}  # (x) = occurrences
    tag_unigrams = {}   # (POS) = occurrences
    pos_unigrams = {}   # (x,POS) = occurrences
    words = []

    tokenCount = 0


    # Each line is a sentence:
    # ['<s>', 'I/NN', 'am/VBD', 'happy/ADJ', '</s>']
    sentences = []
    f = open(trainingFile, 'r')
    for line in f:
        line = line.rstrip('\n')
        sentence = ['<s>/<s>']
        sentence += line.split(" ")
        sentence += ["</s>"]
        sentences += sentence

    # Capture words on their own
    for word in sentences:
        tokenCount += 1
        if word == "</s>":
             x = "</s>"
             x_POS = "</s>"
        else:
            splitWord = word.split('/')
            if len(splitWord) < 3:
                x = splitWord[0]
                x_POS = splitWord[1]
            else:
                x = splitWord[1]
                x_POS = splitWord[2]
        # get (x,x_POS)
        if (x,x_POS) in pos_unigrams.keys(): pos_unigrams[(x,x_POS)] += 1
        else: pos_unigrams[(x,x_POS)] = 1
        # get (POS)
        if x_POS in tag_unigrams.keys(): tag_unigrams[x_POS] += 1
        else: tag_unigrams[x_POS] = 1
        # get (x)
        if x in word_unigrams.keys(): word_unigrams[x] += 1
        else: word_unigrams[x] = 1

    # Capture words in pairs
    for i in range(0, len(sentences)-1):
        word1 = sentences[i].split('/')
        word2 = sentences[i+1].split('/')
        if sentences[i] == "</s>": word1 = ["</s>", "</s>"]
        if sentences[i+1] == "</s>": word2 = ["</s>", "</s>"]
        if len(word1) > 2:
            word1[0],word1[1] = word1[1], word1[2]
        if len(word2) > 2:
            word2[0],word2[1] = word2[1], word2[2]
        # fill up dictionaries
        if (word1[1],word2[1]) in tag_bigrams.keys():
            tag_bigrams[(word1[1],word2[1])] += 1
        else:
            tag_bigrams[(word1[1],word2[1])] = 1

    # write transitions.txt
    with open('transitions.txt', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        writer.writerow(['POS(x)', 'POS(y)', 'P_MLE(POS(y)|POS(x))'])
        for (tag1,tag2), count in tag_bigrams.items():
            # P_MLE = (x,y occurences) / (x occurences)
            MLE = count / tag_unigrams[tag1]
            writer.writerow([tag1, tag2, MLE])

    # write emissions.txt
    with open('emissions.txt', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        writer.writerow(['POS(x)', 'x', 'P_MLE(x|POS(x))', 'P_Laplace(x|POS(x))',
            'P_MLE(POS(x)|x)', 'P_Laplace(x|POS(x))'])
        for (x,x_POS), count in pos_unigrams.items():
            # P_MLE = (x has tag t) / (tag t occurs)
            MLE = (count) / (tag_unigrams[x_POS])
            MLE_r = (count) / (word_unigrams[x])
            # P_Laplace = P(y|x) = #(x,xPOS)+1 / (#(x) + vocab size + 1)
            Laplace = (count + 1) / (tag_unigrams[x_POS] + len(tag_unigrams) + 1)
            Laplace_r = (count + 1) / (word_unigrams[x] + len(word_unigrams) + 1)
            writer.writerow([x_POS, x, MLE, Laplace, MLE_r, Laplace_r])
        for x in tag_unigrams.keys():
            Laplace = 1 / (tag_unigrams[x] + len(word_unigrams) + 1)
            writer.writerow([x, 'unk', 0.0, Laplace, 0.0, 0.0])

    # write laplace-tag-unigrams.txt
    with open('laplace-tag-unigrams.txt', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        writer.writerow(['POS(x)', 'P_Laplace(POS(x))'])
        for x_POS, count in tag_unigrams.items():
            # P_Laplace for unigrams
            Laplace = (tag_unigrams[x_POS] + 1) / (tokenCount + len(word_unigrams) + 1)
            writer.writerow([x_POS, Laplace])
