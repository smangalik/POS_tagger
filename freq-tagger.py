import sys
import csv
import string
from collections import defaultdict

if __name__ == '__main__':
    # intialize core variables
    if len(sys.argv) == 4:
        testFile = str(sys.argv[1])
        emissionsFile = str(sys.argv[2])
        outputFile = str(sys.argv[3])
    else:
        print("<Using Default Parameters>")
        testFile = 'test.txt'
        emissionsFile = 'emissions.txt'
        outputFile = 'test-fbtagger.txt'

    tagUnigramFile = 'laplace-tag-unigrams.txt' # TODO confirm
    frequency_predictions = {}   # (x) = (POS(x), count) [GUESS]
    tag_probability = {}         # (POS) = P_Laplace(POS)
    default_prediction = 'TAG'


    # read tag-unigrams
    with open(tagUnigramFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = False
        for row in csvreader:
            if headerCaught == False:
                headerCaught = True
                continue
            tag, probability = row
            # Add probability to dictionary
            tag_probability[tag] = probability
    default_prediction = max(tag_probability, key=tag_probability.get)


    # read emissions
    with open(emissionsFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = False
        for row in csvreader:
            if headerCaught == False:
                headerCaught = True
                continue

            x_POS, x, MLE, Laplace, MLE_r, Laplace_r = row
            # Update prediction
            if x in frequency_predictions.keys():
                # Better POS tag
                if MLE_r > frequency_predictions[x][1]:
                    frequency_predictions[x] = (x_POS, MLE_r)
                # Tied POS tag
                if MLE_r == frequency_predictions[x][1]:
                    oldTag = frequency_predictions[x][0]
                    newTag = x_POS
                    if tag_probability[newTag] > tag_probability[oldTag]:
                        frequency_predictions[x] = (newTag, MLE_r)
            # Create prediction
            else:
                frequency_predictions[x] = (x_POS, MLE_r)

    totalPredictions = 0
    correctPredictions = 0
    tag_stats = {} # POS = [correctPredict, systemPredict, testCount]

    # write output-fbtagger.txt
    test = open(testFile, 'r')
    output = open(outputFile, 'w')
    for line in test:
        line = line.rstrip('\n')
        words = line.split(' ')
        for word in words:
            split = word.split('/')
            # taggable
            if len(split) >= 2:
                tag = split[-1]
                content = split[-2]
                if len(split) > 2: content = '/'.join(split[0:-1])
                if content in frequency_predictions.keys():
                    determinedTag = frequency_predictions[content][0]
                else: determinedTag = default_prediction
                output.write(content +'/' + tag + '/' + determinedTag + ' ')
                # ERROR ANALYSIS
                if tag not in tag_stats.keys(): tag_stats[tag] = [0,0,0]
                if determinedTag not in tag_stats.keys(): tag_stats[determinedTag] = [0,0,0]
                if tag == determinedTag:
                    correctPredictions += 1
                    tag_stats[tag][0] += 1
                tag_stats[determinedTag][1] += 1
                tag_stats[tag][2] += 1
                totalPredictions += 1

            # untaggable
            else:
                output.write(word + ' ')
        output.write('\n')

    accuracy = correctPredictions/totalPredictions
    print("Overall accuracy for freq-tagger is: " + str(accuracy))
    for tag, stats in tag_stats.items():
        if stats[1] == 0: precision = 0
        else: precision = round(stats[0]/stats[1],3)
        if stats[2] == 0: recall = 0
        else: recall = round(stats[0]/stats[2],3)
        if precision == 0 or recall == 0: F1 = 0
        else: F1 = round((2 * precision * recall)/(precision + recall),3)
        print(tag,'\t(precision:',str(precision),')(recall:',recall,')(F1:',F1,')')
