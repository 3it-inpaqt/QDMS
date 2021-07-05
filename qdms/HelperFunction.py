import datetime

# GLOBAL_ID is a unique id per program run
GLOBAL_ID = datetime.datetime.now()
path = './/Simulation//'+GLOBAL_ID.strftime("%Y_%m_%d-%Hh_%Mm_%Ss")
# directoryname = 'C:\\Users\\Sebastien Graveline\\Desktop\\'+GLOBAL_ID.strftime("%Y_%m_%d-%Hh_%Mm_%Ss")


def is_square(positive_int):
    """
    Quick function to find if a number is a perfect square root

    Parameters
    ----------
    positive_int : int
        The number evaluated.

    Returns
    ----------
    bool : bool
        If true, the number is a perfect square root.

    """
    x = positive_int // 2
    seen = set([x])
    while x * x != positive_int:
        x = (x + (positive_int // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


def limit_vector(vector,  bottom_limit, upper_limit):
    """
    This function cut the a vector to keep only the values between the bottom_limit and the upper_limit.

    Parameters
    ----------
    vector : list
        The vector that will be cut

    upper_limit : float
        The maximum value of the vector.

    bottom_limit : float
        The minimum value of the vector

    Returns
    -------
    vector : iterable
        The limited vector
    """
    temp = []
    for x in vector:
        if bottom_limit <= x <= upper_limit:
            temp.append(x)
    return temp


def simplify_vector_length(vector, final_length):
    """
    Function to resize a vector to a final_length.
    Speed test result in 5,41s for len(vector) = 100 000 000

    Parameters
    ----------
    vector : iterable
        The vector to be simplify.

    final_length : int
        The length of the return vector.

    Returns
    ----------
    simplified_vector : list
        vector with final_length length.
    """

    diff = len(vector) - final_length
    if diff <= 0:
        return vector

    # Skim the middle
    diff = len(vector) - final_length
    number = final_length / diff
    simplified_vector = [vector[0]]
    counter = 0
    for p in range(len(vector)):
        if counter >= number:
            counter -= number
            continue
        simplified_vector.append(vector[p])
        counter += 1
    if simplified_vector[-1] != vector[-1]:
        simplified_vector.append(vector[-1])
    return simplified_vector


def simplify_vector_resolution(vector, step_min):
    """
    Function to keep only the values that respect the resolution in vector.

    Parameters
    ----------
    vector : iterable
        The vector to be simplify.

    step_min : float
        The minimal step between two values.

    Returns
    ----------
    simplified_vector : iterable
        vector with the correct resolution.
    """
    previous = vector[0]-step_min-1
    simplified_vector = []
    for current in vector:
        resolution = current - previous
        if resolution >= step_min:
            simplified_vector.append(current)
            previous = current
            # # Cut values to close of each other
            # distinct = []
            # diff_distinct = np.diff(vector)
            # min_res = np.mean(diff_distinct) * 0.0001
            # for i in range(len(diff_distinct)):
            #     if diff_distinct[i] > min_res:
            #         distinct.append(vector[i])
            # distinct.append(vector[-1])

    return simplified_vector
