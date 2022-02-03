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
    trellis_diagram = np.zeros((2,len(O)))
    for obsr in range(len(O)):
        for i in range(2):
            if obsr == 0:
                trellis_diagram[i][obsr] = pi[i] * B[i][O[obsr]]
            else:
                trellis_diagram[i][obsr] = trellis_diagram[0][obsr-1] * A[0][i] * B[i][O[obsr]] + trellis_diagram[1][obsr-1] * A[1][i] * B[i][O[obsr]]
    return np.sum(trellis_diagram[:,(len(O)-1)]), trellis_diagram


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
    trellis_diagram = np.zeros((2,len(O)))
    most_probable_state_sequence = []
    for obsr in range(len(O)):
        if obsr == 0:
            for i in range(2):
                trellis_diagram[i][obsr] = pi[i] * B[i][O[obsr]]
            path = np.argmax(trellis_diagram[:,obsr])
            most_probable_state_sequence.append(path)
        else:
            path = np.argmax(trellis_diagram[:,obsr-1])
            trellis_diagram[path,obsr] = trellis_diagram[path][obsr-1] * A[path][path]*B[path][O[obsr]]
            trellis_diagram[int(not path),obsr] = trellis_diagram[path][obsr-1] * A[path][int(not path)]*B[int(not path)][O[obsr]]
            most_probable_state_sequence.append(np.argmax(trellis_diagram[:,obsr]))
    return most_probable_state_sequence, trellis_diagram