
def get_top_two(vec):
    """ Return the two largest items in the sequence. The sequence must contain at least two items.
    """
    if vec[0] > vec[1]:
        largest = vec[0]
        largest_idx = 0
        second_largest = vec[1]
        second_largest_idx = 1
    else:
        largest = vec[1]
        largest_idx = 1
        second_largest = vec[0]
        second_largest_idx = 0
    for idx, item in enumerate(vec):
        if item > largest:
            second_largest = largest
            second_largest_idx = largest_idx
            largest = item
            largest_idx = idx
        elif largest > item > second_largest:
            second_largest = item
            second_largest_idx = idx
    return (largest_idx, largest), (second_largest_idx, second_largest)


def n2p(n):
    s = sum(n)
    if s <= 0:
        raise Exception("n2p received bad vector: {}".format(n))
    p = [i/s for i in n]
    return p
