import numpy as np
from hmm import forward, viterbi

A = np.asarray([[0.3, 0.7],
                [0.8, 0.2]])

B = np.asarray([[0.5, 0.3, 0.2],
                [0.1, 0.1, 0.8]])

pi = np.asarray([0.6, 0.4])

O = np.asarray([1, 2, 1, 0])

alpha_gt = np.asarray([[0.18, 0.0172, 0.027276, 0.0054306],
                       [0.04, 0.1072, 0.003348, 0.00197628]])
forward_result_gt = 0.00740688

delta_gt = np.asarray([[0.18, 0.0108, 0.024192, 0.0036288],
                       [0.04, 0.1008, 0.002016, 0.00169344]])
viterbi_result_gt = np.asarray([0, 1, 0, 0])

# Testing
forward_result, alpha = forward(A, B, pi, O)
print('Forward result test: {}'.format(abs(forward_result_gt - forward_result) < 10 ** -5))
print('Forward alpha test: {}'.format(np.all(np.abs(alpha - alpha_gt) < 10 ** -5)))

viterbi_result, delta = viterbi(A, B, pi, O)
print('Viterbi result test: {}'.format((viterbi_result_gt == viterbi_result).all()))
print('Viterbi delta test: {}'.format(np.all(np.abs(delta - delta_gt) < 10 ** -5)))
