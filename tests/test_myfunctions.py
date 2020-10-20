from submodlib import myfunctions
def test_squareInt():
    assert myfunctions.square(5) == 25

def test_squareFloat():
    assert myfunctions.square(5.0) == 25

def test_squareNegative():
    assert myfunctions.square(-5.0) == 25
