import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    trellis_diagram = np.zeros((len(pi),len(O))) #creating a trellis diagram dynamicly depending on state numbers and observation length
    for obsr in range(len(O)):
        for i in range(len(pi)):
            if obsr == 0:#first column in trellis diagram is not depending any previous state, therefore it is sperated with if
                trellis_diagram[i][obsr] = pi[i] * B[i][O[obsr]]
            else:
                trellis_diagram[i][obsr] = trellis_diagram[0][obsr-1] * A[0][i] * B[i][O[obsr]] + trellis_diagram[1][obsr-1] * A[1][i] * B[i][O[obsr]] # by using correct state changes and observation probabilities added with prev. state, current state probability is written to trellis
    return np.sum(trellis_diagram[:,(len(O)-1)]), trellis_diagram #return the last column of trellis and sum(rectation.pdf, pg. 15)


def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    trellis_diagram = np.zeros((len(pi),len(O)))#creating a trellis diagram dynamicly depending on state numbers and observation length
    most_probable_state_sequence = [] #most probable state sequence stored in a list
    for obsr in range(len(O)):
        if obsr == 0:
            for i in range(len(pi)):
                trellis_diagram[i][obsr] = pi[i] * B[i][O[obsr]] #first column in trellis diagram is not depending any previous state, therefore it is sperated with if
            path = np.argmax(trellis_diagram[:,obsr]) # take the most probable state as path
            most_probable_state_sequence.append(path)
        else:
            path = np.argmax(trellis_diagram[:,obsr-1]) #from the last observation, get the most probable 
            for j in range(len(pi)):
                trellis_diagram[j][obsr] = trellis_diagram[path][obsr-1] * A[path][j] * B[j][O[obsr]] # calculate the states
            most_probable_state_sequence.append(np.argmax(trellis_diagram[:,obsr])) # save the most probable path  (rectation.pdf, pg. 26)
    return most_probable_state_sequence, trellis_diagram