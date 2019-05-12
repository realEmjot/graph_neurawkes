from collections import Counter
from itertools import product

import numpy as np
from scipy.spatial import distance


def calculate_jaccard_index(seq1, seq2):
    uniq1 = set(seq1)
    uniq2 = set(seq2)
    return len(uniq1 & uniq2) / len(uniq1 | uniq2)


def calculate_multiset_jaccard_index(seq1, seq2):
    c1 = Counter(seq1)
    c2 = Counter(seq2)

    intersection_count = 0
    union_count = 0

    for key in (set(c1.keys()) | set(c2.keys())):
        count1 = c1[key]
        count2 = c2[key]

        intersection_count += min(count1, count2)
        union_count += max(count1, count2)

    return intersection_count / union_count


def _calculate_jensen(seq1, seq2, num_types):
    p1 = np.bincount(seq1, minlength=num_types)
    p1 = p1 / p1.sum()

    p2 = np.bincount(seq2, minlength=num_types)
    p2 = p2 / p2.sum()
    
    return distance.jensenshannon(p1, p2)

####################################################

def calculate_sender_or_recipient_jaccard(seq1, seq2, jacc_func=calculate_jaccard_index):
    return jacc_func(
        [e[0] for e in seq1] + [e[1] for e in seq1],
        [e[0] for e in seq2] + [e[1] for e in seq2]
    )


def calculate_sender_or_recipient_jensen(seq1, seq2, num_types):
    return _calculate_jensen(
        [e[0] for e in seq1] + [e[1] for e in seq1],
        [e[0] for e in seq2] + [e[1] for e in seq2],
        num_types
    )


def calculate_sender_jaccard(seq1, seq2, jacc_func=calculate_jaccard_index):
    return jacc_func(
        [e[0] for e in seq1],
        [e[0] for e in seq2]
    )


def calculate_sender_jensen(seq1, seq2, num_types):
    return _calculate_jensen(
        [e[0] for e in seq1],
        [e[0] for e in seq2],
        num_types
    )


def calculate_recipient_jaccard(seq1, seq2, jacc_func=calculate_jaccard_index):
    return jacc_func(
        [e[1] for e in seq1],
        [e[1] for e in seq2]
    )


def calculate_recipient_jensen(seq1, seq2, num_types):
    return _calculate_jensen(
        [e[1] for e in seq1],
        [e[1] for e in seq2],
        num_types
    )


def calculate_edge_jaccard(seq1, seq2, jacc_func=calculate_jaccard_index):
    return jacc_func(
        [e[:2] for e in seq1],
        [e[:2] for e in seq2]
    )


def calculate_edge_jensen(seq1, seq2, num_types):
    return _calculate_jensen(
        [num_types * s + r for s, r, _ in seq1],
        [num_types * s + r for s, r, _ in seq2],
        num_types
    )


def calculate_full_jaccard(seq1, seq2):
    return calculate_jaccard_index(seq1, seq2)



def calculate_time_deltas_distribution(seq1, seq2):
    return np.histogram([e1[2] - e2[2] for e1, e2 in zip(seq1, seq2)])


seq1 = [
    (1, 2, 0.4),
    (0, 1, 0.6),
    (0, 2, 0.85)
]
seq2 = [
    (0, 1, 0.3),
    (0, 1, 0.6),
    (1, 1, 0.9)
]

# print(calculate_sender_or_recipient_jaccard(seq1, seq2))
# print(calculate_sender_jaccard(seq1, seq2, calculate_multiset_jaccard_index))
# print(calculate_recipient_jaccard(seq1, seq2, calculate_multiset_jaccard_index))
#print(calculate_time_deltas_distribution(seq1, seq2))
print(calculate_sender_or_recipient_jensen(seq1, seq2, 3))