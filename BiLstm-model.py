#!/usr/bin/env python3

"""
LSTM & double linear Network
"""

# import torch
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import re
import string
from config import device

"""
Yang Zhen z5210646 & Ziyan Chen z5210614

For preprocessing, we remove the string punctraion, non-acscii charater and map number to word.
Stopwords are copied from library and will be removed.

We choose word vector with 300 dimensions, as in our testing the higher the dimension is, the better the result.

For network structure: we chose bidirectional LSTM with 100 hidden nodes and 2 LSTM layer (bidiretional) with a dropout of 5.
Then we attach two fully connected layers with Relu as activation function. We have two models, one for rating with 1 output and one
for catagery with 5 outputs. Different approach: 
    more hidden lstm nodes -> longer training, not benifcical to outcome
    more hidden fully connected nodes -> longer training, not too much difference to the outcome
    more layers -> converge at a lower accuracy (75%)
    more fully conected layers -> longer training, slightly worse outcome (79%)
    one model with 7 outputs, two for rating and five for classification -> no difference
    one model with 10 outputs (2*5) -> random behaviour be genrally worse than current approach
    rating model with 2 outputs -> slightly worse than one output with bce as loss function

loss funtion: we use bse for rating and cross entropy for classfication, return 0.4 * rating loss and 0.6 * classificaton lost,
this perfroms better than 1/2 for each, and sometime being too partial to classificaton lost might to lead to totally ignorance of
rating loss (this happended when we assign 0.33 for rating loss)

Epochs: 5 epochs, small network with simple structure. Only needs 3-5 epochs to train to optimal

Optimizer : we use Adam as optimizer since Adam is better at converging speed, we tried weigth decay which is pretty worse in out case

Our failed (or worse in some way) model:
CNNs -> LSTM -> fully connected, CNNs has window size 1,3,5. Aiming to get the meaning for 1,3,5 words when processing. Doing slightly
worse than current approach, plus it's slower to train

LSTM -> CNN -> fully connected, Slightly worse (80%)

CNN -> LSTM -> CNN -> fully connected. Just not working very well.
"""

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    sample = sample.lower()
    for a in "!#$%&()*'+,-./:;<=>?@[\]^_`{|}~":      #remove punctration
        sample = sample.replace(a,"")
    sample = sample.strip()        #remove trailling space
    processed = sample.split()

    return processed

def preprocessing(sample):
    number_dic = {'0': "zero ", '1': "one", '2': "two", '3': "three", '4': "four", '5': "five", '6': "six",
                    '7': "seven", '8': "eight", '9': "nine", '10' : "ten"} # mapping number into words
    sample = [re.sub(r'[^\x00-\x7f]', r'', w) for w in sample]    #remove illegal character
    temp = []
    for i in sample:
        if i not in stopWords:
            i = re.sub(r"^goo[o]+d$", "good", i)   # just want to normalize good
            if i in number_dic.keys():
                i = number_dic[i]
            i = re.sub(r"\d+", "", i) # remove any remaing digits
            temp.append(i)
    return temp

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {
    "a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", 
    "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", 
    "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", 
    "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", 
    "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", 
    "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", 
    "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", 
    "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", 
    "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", 
    "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", 
    "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", 
    "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", 
    "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", 
    "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", 
    "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", 
    "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", 
    "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", 
    "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", 
    "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", 
    "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", 
    "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", 
    "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", 
    "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", 
    "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", 
    "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep	keeps", "kept", 
    "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", 
    "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", 
    "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", 
    "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", 
    "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", 
    "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", 
    "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", 
    "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", 
    "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", 
    "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", 
    "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", 
    "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", 
    "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", 
    "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", 
    "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", 
    "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", 
    "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", 
    "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", 
    "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", 
    "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", 
    "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", 
    "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", 
    "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", 
    "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", 
    "such", "sufficiently", "suggest", "sup", "sure	t", "take", "taken", "taking", "tell", "tends", 
    "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", 
    "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", 
    "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", 
    "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", 
    "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", 
    "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", 
    "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", 
    "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", 
    "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", 
    "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", 
    "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", 
    "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", 
    "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", 
    "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", 
    "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", 
    "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", 
    "yourself", "yourselves", "you've", "z", "zer", ""
}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """


    m = tnn.Sigmoid()
    ratingOutput = torch.round(m(ratingOutput))
    ratingOutput = ratingOutput.long()
    categoryOutput = torch.argmax(categoryOutput, dim = -1)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """
    # LSTM -> fully connected -> Relu -> fully connected
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(300, 200, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.catagrey_body = tnn.Sequential(
            tnn.Linear(400, 64),
            tnn.ReLU(),
            tnn.Linear(64, 5) # 5 catagery
        )
        self.rating_body = tnn.Sequential(
            tnn.Linear(400, 64),
            tnn.ReLU(),
            tnn.Linear(64, 1) # determing positive or negative, one output performs better than two (sligtly)
        )

    def forward(self, input, length):
        out, _ = self.lstm(input)
        out = out[:, -1, :]
        rating = self.rating_body(out)
        categary = self.catagrey_body(out)
        rating = rating.squeeze()
        return rating, categary

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.cross = tnn.CrossEntropyLoss()  # crossentropy for classificaiton
        self.bce = tnn.BCEWithLogitsLoss()   # bce for negative or possitive

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingTarget = ratingTarget.float()
        a = self.bce(ratingOutput, ratingTarget)
        b = self.cross(categoryOutput, categoryTarget)
        return a/2 + b/2

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.95 # getting more examples to train
batchSize = 64
epochs = 9

optimiser = toptim.Adam(net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08) # low learning rate