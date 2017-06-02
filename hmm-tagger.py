import sys
import csv
import string
import numpy as np

def queryA(POS_y,POS_x):
    # if A[POS,POS] fails, return 0
    if (POS_x,POS_y) in A.keys():
        return A[(POS_x,POS_y)]
    else:
        return .000000000000000000000000000001

def queryB(POS_x,x):
    # if B[x,POS] fails, return B['unk',POS]
    if (x,POS_x) in B.keys():
        return B[(x,POS_x)]
    elif M_L == 'M': return .000000000000000000000000000001
    else: return B[('unk',POS_x)]

def Viterbi(S):
    finalProbability = 0
    bestTags = []
    n = len(S)
    T =  list(tags.keys())
    delta = [[0 for x in range(len(T))] for y in range(n)]
    psi = [[0 for x in range(len(T))] for y in range(n)]

    # initialize
    for i in range(len(T)):
        aHere = float(queryA('<s>',T[i]))
        bHere = float(queryB(T[i],S[0]))
        if aHere == 0: aHere = .000000000000000000000000000001
        if bHere == 0: bHere = .000000000000000000000000000001
        entry = np.log2(aHere) + np.log2(bHere)
        #entry = aHere + bHere
        delta[0][i] = entry
        psi[0][i] = T.index("<s>")


    # forward pass: use sum_of_logs instead of product of probabilities
    for i in range(1,n):
        for j in range(len(T)):
            deltas = []
            bestPsi = 0
            bestPsiValue = float('-inf')
            bHere = float(queryB(T[j],S[i]))
            for k in range(len(T)):
                aHere = float(queryA(T[k],T[j]))
                if aHere == 0: aHere = .000000000000000000000000000001
                if bHere == 0: bHere = .000000000000000000000000000001
                deltaEntry = delta[i-1][k] + np.log2(aHere) + np.log2(bHere)
                #deltaEntry = delta[i-1][k] * aHere * bHere
                deltas.append(deltaEntry)
                if(deltaEntry > bestPsiValue): # better tag
                    bestPsiValue = deltaEntry
                    bestPsi = k
                elif(deltaEntry == bestPsiValue): # tied tag
                    if tags[T[k]] > tags[T[bestPsi]]:
                        bestPsi = k
            delta[i][j] = bestPsiValue
            psi[i][j] = bestPsi

    # back trace
    maxTag = float('-inf')
    maxarg = 0
    for x in range(len(T)):
        deltaValue = delta[n-1][x]
        if(maxTag < deltaValue):
            maxTag = deltaValue
            maxarg = x
    tagrev = []
    for x in range(n-1, -1, -1):
        tagrev.append(maxarg)
        maxarg = psi[x][maxarg]
    bestTags = list(reversed(tagrev))
    bestTags[0] = T.index('<s>')
    bestTags[-1] = T.index('</s>')

    # TODO check back tracing

    # extract POS tags
    tagArray = [T[i] for i in bestTags]

    # determine final probability
    for i in range(1,n):
        P_emission = float(queryB(T[bestTags[i]],tagArray[i]))
        P_transition = float(queryA(tagArray[i],tagArray[i-1]))
        finalProbability = P_emission * P_transition
        finalProbability = round(finalProbability, 4)

    #print(tagArray)
    #print(S)
    #print(finalProbability)
    #exit()

    return [finalProbability,tagArray]

if __name__ == '__main__':
    # intialize core variables
    if len(sys.argv) == 7:
        M_L = str(sys.argv[1]).upper()
        testFile = str(sys.argv[2])
        transitionsFile = str(sys.argv[3])
        emissionsFile = str(sys.argv[4])
        tagUnigramFile = str(sys.argv[5])
        outputFile = str(sys.argv[6])
        if M_L != 'M' or 'L':
            print("You did not pick a proper model (M or L)")
    else:
        print("<Using Default Parameters>")
        M_L = 'L'
        testFile = 'test.txt'
        transitionsFile = 'transitionsJ.txt' #reset
        emissionsFile = 'emissions.txt'
        tagUnigramFile = 'laplace-tag-unigrams.txt'
        if M_L == 'M': outputFile = 'test-hmm-mle-tagger.txt'
        if M_L == 'L': outputFile = 'test-hmm-laplace-tagger.txt'

    # Key Variables
    A = {} # (POS->POS) = transitions
    B = {} # (x,POS) = emissions
    tags = {} # (tag) = probability
    S = [] # sentences_i 0-199
    T = [] # tagSequence_i 0-199
    totalPredictions = 0
    correctPredictions = 0
    tag_stats = {} # POS = [correctPredict, systemPredict, testCount]

    # read tag-unigrams
    with open(tagUnigramFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = False
        for row in csvreader:
            if headerCaught == False:
                headerCaught = True
                continue
            # Add probability to dictionary
            tag, probability = row
            tags[tag] = probability

    # Populate A from transitions
    with open(transitionsFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = True # TODO remove
        for row in csvreader:
            if headerCaught == False:
                headerCaught = True
                continue
            #if len(row) != 3: print(row)
            POS_x, POS_y, P_xy = row
            # Add probability to dictionary
            A[(POS_x,POS_y)] = P_xy

    # Populate B from emissions
    with open(emissionsFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = False
        for row in csvreader:
            if headerCaught == False:
                headerCaught = True
                continue
            if len(row) != 6: print(len(row),row)
            # Add probability to dictionary
            x_POS, x, MLE, Laplace, MLE_r, Laplace_r = row
            #x, x_POS, MLE, Laplace, MLE_r, Laplace_r = row
            if M_L == 'M':
               B[(x,x_POS)] = MLE_r
            elif M_L == 'L':
                B[(x,x_POS)] = Laplace_r

    # Each line is a sentence:
    f = open(testFile, 'r')
    for line in f:
        line = line.rstrip('\n')
        tagsequence = ['<s>']
        sentence = ['<s>']
        for word in line.split(" "):
            sentence += [word.split('/')[0]]
            tagsequence += [word.split('/')[-1]]
        sentence += ["</s>"]
        tagsequence += ["</s>"]
        S.append(sentence)
        T.append(tagsequence)

    # Output Model
    output = open(outputFile, 'w')
    for i in range(len(S)):
        sentence = S[i]
        actualTags = T[i]

        # run viterbi algo
        result = Viterbi(sentence)
        predictedScore = result[0]
        predictedTags = result[1]

        # iterate over sentence - write to output + collect stats
        output.write(str(predictedScore))
        output.write('\t')
        for j in range(len(sentence)):
            actualTag = actualTags[j]
            determinedTag = predictedTags[j]
            totalPredictions += 1
            if actualTag not in tag_stats.keys(): tag_stats[actualTag] = [0,0,0]
            if determinedTag not in tag_stats.keys(): tag_stats[determinedTag] = [0,0,0]
            if actualTag == determinedTag:
                correctPredictions += 1
                tag_stats[actualTag][0] += 1
            tag_stats[determinedTag][1] += 1
            tag_stats[actualTag][2] += 1
            line = sentence[j] + '/' + actualTag + '/' + determinedTag + ' '
            output.write(line)
        output.write('\n')

    # Output final probabilities
    if M_L == 'M': print('Using MLE emissions')
    if M_L == 'L': print('Using Laplace emissions')
    accuracy = round(correctPredictions/totalPredictions, 4) * 100
    print("Overall accuracy for hmm-tagger is: " + str(accuracy),"%")
    for tag, stats in tag_stats.items():
        if stats[1] == 0: precision = 0
        else: precision = round(stats[0]/stats[1],3)
        if stats[2] == 0: recall = 0
        else: recall = round(stats[0]/stats[2],3)
        if precision == 0 or recall == 0: F1 = 0
        else: F1 = round((2 * precision * recall)/(precision + recall),3)
        print(tag,'\t(precision:',str(precision),')(recall:',recall,')(F1:',F1,')')
