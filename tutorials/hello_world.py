import submodlib #Pure Python modules
import submodlib_cpp #CPP Engine backed modules
from submodlib import myfunctions 
from submodlib_cpp import mess 

x = input("Please enter a number whose square you want:")
x_square = myfunctions.square(float(x))
print("The square of " + x + " = " + str(x_square))

s = input("Please enter a message to print:")
obj = mess(s)
obj.out()
