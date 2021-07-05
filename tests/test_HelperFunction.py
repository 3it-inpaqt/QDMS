import qdms


def test_limit_vector():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert qdms.HelperFunction.limit_vector(x, 2, 7) == [2, 3, 4, 5, 6, 7]


def test_simplify_vector_length():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = qdms.HelperFunction.simplify_vector_length(x, 4) == [0, 0, 2, 5, 7, 9]
    assert result


def test_simplify_vector_resolution():
    x = [0, 1, 1.1, 1.3, 4, 4.1, 4.3]
    result = qdms.HelperFunction.simplify_vector_resolution(x, 0.2) == [0, 1, 1.3, 4, 4.3]
    assert result