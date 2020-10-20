import submodlib
from submodlib import myfunctions

x = input("Please enter a number whose square you want:")
x_square = myfunctions.square(float(x))
print("The square of " + x + " = " + str(x_square))