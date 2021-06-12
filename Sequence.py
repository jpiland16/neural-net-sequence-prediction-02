import random, pickle

class Sequence():
    def __init__(self, seq: list, unique_items: int) -> None:
        self.list = seq
        self.unique_items = unique_items

    def at(self, pos: int) -> int:
        """
        Return the value at the specified position in the list. Can be
        negative or even outside the range of the list, i.e., out-of-bounds
        indices will loop back to the beginning of the list.
        """
        return self.list[pos % len(self.list)]

    def subseq_from(self, start: int, length: int) -> 'Sequence':
        """
        Get a subseqence of the specified length from this sequence.
        Loops around back to the beginning to prevent NaN returns.
        """
        subseq = Sequence([], self.unique_items)
        for i in range(length):
            subseq.list.append(self.at(start + i))
        return subseq

    def __str__(self):
        return f'Sequence.Sequence({str(self.list)}, {self.unique_items})'

    def __repr__(self):
        return str(self)


def generate_sequence(unique_items: int, length: int) -> Sequence:
    """
    Generate a sequence with the specified length and number of
    unique items, and return the sequence as a Sequence object.

    Also saves the sequence to disk for later retrieval.
    """
    seq = Sequence([], unique_items)

    for _ in range(length):
        seq.list.append(random.randint(0, unique_items - 1))

    with open('seq.pkl', 'wb') as file:
        pickle.dump(seq, file)

    return seq

def get_sequence() -> Sequence:
    """
    Return the sequence that was saved on disk, if it exists.
    """
    try:
        pkl = open('seq.pkl', 'rb')
        seq = pickle.load(pkl)
        return seq
    except:
        return None
        # raise FileNotFoundError
