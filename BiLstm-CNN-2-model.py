#!/usr/bin/env python3

"""
Bi-LSTM & single CNN & double linear
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

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    sample = sample.lower()
    for a in "!#$%&()*+,-./:;<=>?@[\]^_`{|}~":
        sample = sample.replace(a,"")
    sample = sample.strip()
    processed = sample.split()

    return processed

def preprocessing(sample):
    sample = [re.sub(r'[^\x00-\x7f]', r'', w) for w in sample]
    number_dic = {'0': "zero ", '1': "one", '2': "two", '3': "three", '4': "four", '5': "five", '6': "six",
                    '7': "seven", '8': "eight", '9': "nine", '10' : "ten"}
    temp = []
    for i in sample:
        if i not in stopWords:
            i = re.sub(r"^goo[o]+d$", "good", i)
            if i in number_dic.keys():
                i = number_dic[i]
            i = re.sub(r"\d+", "", i)
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
    ratingOutput = torch.argmax(ratingOutput,dim = -1)
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

    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(300, 100, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.catagrey_body1 = tnn.Sequential(
            tnn.Conv1d(1,35,21),
            tnn.ReLU(),
            tnn.MaxPool1d(9,9),
            # tnn.Conv1d(10,50,21),
            # tnn.ReLU(),
            # tnn.MaxPool1d(4,4),
        )
        self.catagrey_body2 = tnn.Sequential(
            tnn.Linear(700, 128),
            tnn.ReLU(),
            tnn.Linear(128, 5)
        )
        self.rating_body1 = tnn.Sequential(
            tnn.Conv1d(1,35,21),
            tnn.ReLU(),
            tnn.MaxPool1d(9,9),
            # tnn.Conv1d(10,50,21),
            # tnn.ReLU(),
            # tnn.MaxPool1d(4,4),
        )
        self.rating_body2 = tnn.Sequential(
            tnn.Linear(700, 128),
            tnn.ReLU(),
            tnn.Linear(128, 2),
        )

    def forward(self, input, length):
        out, _ = self.lstm(input)
        out = out[:, -1, :]
        out1 = out.view(out.shape[0],1,200)
        rating = self.rating_body1(out1)
        rating = rating.view(out1.shape[0],700)
        rating = self.rating_body2(rating)
        categary = self.catagrey_body1(out1)
        categary = categary.view(out1.shape[0],700)
        categary = self.catagrey_body2(categary)
        return rating, categary

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.cross = tnn.CrossEntropyLoss()
        self.bce = tnn.BCEWithLogitsLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        a = self.cross(ratingOutput, ratingTarget)
        b = self.cross(categoryOutput, categoryTarget)
        return a/2 + b/2

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.9
batchSize = 64
epochs = 8
# optimiser = toptim.SGD(net.parameters(), lr=0.01)

optimiser = toptim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)
