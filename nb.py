import numpy as np
import re

def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    return set(np.unique(np.array(list(np.concatenate(np.asarray(data,object)).flat))))

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    pi_dict = {}
    result = np.unique(train_labels,return_counts=True)
    for i in range(len(result[0])):
        pi_dict[result[0][i]] = result[1][i]/np.sum(result[1])
    return pi_dict
    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    from collections import Counter
    result_set = {}
    for class_name in np.unique(train_labels):
        class_indices = np.where(np.array(train_labels) == class_name)[0]
        a = np.array(list(np.concatenate(np.asarray(np.array(train_data,object)[class_indices],object)).flat))
        index, count = np.unique(a, return_counts=True)
        zip_iterator = zip(index, (count))
        a_dictionary = dict(zip_iterator)
        add_dict = Counter(a_dictionary) + Counter(vocab)
        dict_3 = dict(add_dict)
        result = np.array(list(dict_3.values()))/(a.size+len(list(vocab)))
        result_set[class_name] = dict(zip(dict_3.keys(),result))
    return result_set
    
def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    result_set = []
    for data in test_data:
        class_probability = []
        for class_label in list(pi.keys()):
            probability = np.log(pi[class_label])
            for datum in data:
                if(datum in vocab):
                    probability += np.log(theta[class_label][datum])
            result_tuple = (probability, class_label)
            class_probability.append(result_tuple)
        result_set.append(class_probability)
    return result_set

def predict(train_set, train_labels, test_set, test_labels):
    vocab = vocabulary(train_set)
    pi = estimate_pi(train_labels)
    theta = estimate_theta(train_set, train_labels, vocab)
    scores = test(theta, pi, vocab, test_set)


    predictions = []
    for datum_result in np.array(scores):
        prediction = datum_result[:,1][np.argmin(datum_result[:,0])]#argmax uses abs to compare
        predictions.append(prediction)
    return np.array(predictions)

def accuracy(predictions, test_labels):
    return (np.count_nonzero((np.unique(predictions, return_inverse=True)[1] - np.unique(test_labels, return_inverse=True)[1]) == 0 )) / len(test_labels) #converting labels into integers and getting the difference, counting zeros and finding accuracy


def main():
    # i am loading the data line by line as one sentence to each line
    # i am applying a preprocesisng by removing any punctiation from both test and training data. then i splitt the sentence into words
    my_file = open("nb_data/train_set.txt", "r", encoding="utf-8")
    train_set = my_file.read().split("\n")
    dataset = []
    for sentence in train_set:
        dataset.append(re.sub(r'[^\w\s]','',sentence).split())
    my_file.close()
    train_set = dataset

    my_file = open("nb_data/train_labels.txt", "r", encoding="utf-8")
    train_labels = my_file.read().split("\n")


    my_file = open("nb_data/test_set.txt", "r", encoding="utf-8")
    test_set = my_file.read().split("\n")
    dataset = []
    for sentence in test_set:
        dataset.append(re.sub(r'[^\w\s]','',sentence).split())
    my_file.close()
    test_set = dataset

    my_file = open("nb_data/test_labels.txt", "r", encoding="utf-8")
    test_labels = my_file.read().split("\n")

    predictions = predict(train_set, train_labels, test_set, test_labels)
    print(round(accuracy(predictions, test_labels),2))


if __name__ == "__main__":
    main()