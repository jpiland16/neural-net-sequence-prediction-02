from numpy import argmax, array, ndarray
from Sequence import Sequence, get_sequence

def one_hot_encode(seq: Sequence) -> ndarray:
    """
    Returns a `numpy` array of N-element vectors where N is the number of
    unique items in the Sequence. Uses a one-hot encoding to 
    produce sparse, binary vectors.
    """
    encoded_list = []
    for value in seq.list:
        vector = [0 for _ in range(seq.unique_items)]
        vector[value] = 1
        encoded_list.append(vector)
    return array(encoded_list)

def one_hot_decode(encoded_list: ndarray) -> Sequence:
    """
    Return the decoded value for each vector in `encoded_list`,
    i.e., the index with a "1".
    """
    # TODO: delete this method if unused
    seq = Sequence([], len(encoded_list[0]))
    seq.list = [argmax(vector) for vector in encoded_list]
    return seq

def show_seq_str():
    """
    Show the saved sequence.
    """
    seq = get_sequence()
    print(seq)

def confirm(msg: str) -> bool:
    """
    Ask the user for confirmation.
    """
    res = input(msg + " (Y/n) > ")
    if res == 'Y' or res == 'y' or res == 'yes' or res == 'Yes' or res == "":
        return True
    return False
