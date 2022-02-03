from nb import vocabulary, estimate_pi, estimate_theta, test

train_data = [['a', 'b', 'b', 'd'],
              ['a', 'c', 'd'],
              ['a', 'c', 'd', 'c', 'a'],
              ['c', 'c', 'e']]

train_labels = ['class1', 'class1', 'class2', 'class2']

test_data = [['d', 'd', 'a'],
             ['e', 'c', 'c', 'b']]

test_labels = ['class1', 'class2']

vocab_gt = {'d', 'b', 'e', 'a', 'c'}

theta_gt = {
    'class2': {'e': 0.15384615384615385, 'a': 0.23076923076923078, 'c': 0.38461538461538464, 'd': 0.15384615384615385,
               'b': 0.07692307692307693},
    'class1': {'e': 0.08333333333333333, 'a': 0.25, 'c': 0.16666666666666666, 'd': 0.25, 'b': 0.25}}

scores_gt = [[(-5.903088603156555, 'class2'), (-4.852030263919617, 'class1')],
             [(-7.040921604977946, 'class2'), (-8.147867129923947, 'class1')]]

pi_gt = {'class2': 0.5, 'class1': 0.5}

# Testing
vocab = vocabulary(train_data)
print('Vocabulary result: {}'.format(vocab_gt == vocab))

pi = estimate_pi(train_labels)
theta = estimate_theta(train_data, train_labels, vocab)
theta_success = True
for class_name in theta_gt:
    for word in theta[class_name]:
        if abs(theta[class_name][word] - theta_gt[class_name][word]) > 10 ** -5:
            theta_success = False
print('Theta result: {}'.format(theta_success))
pi_success = True
for class_name in pi_gt:
    if abs(pi_gt[class_name] - pi[class_name]) > 10 ** -5:
        pi_success = False
print('Pi result: {}'.format(pi_success))

scores = test(theta, pi, vocab, test_data)
scores_success = True
for score_gt, score_pred in zip(scores_gt, scores):
    gt_dict = dict([(x[1], x[0]) for x in score_gt])
    pred_dict = dict([(x[1], x[0]) for x in score_pred])
    for class_name in gt_dict:
        if abs(gt_dict[class_name] - pred_dict[class_name]) > 10 ** -5:
            scores_success = False
print('Scores result: {}'.format(scores_success))
