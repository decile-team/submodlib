#In this implementation of packaging (using CMake) we can import
#both cpp backed modules and actual python modules from
#same library name (here submodlib_alt)
from submodlib_alt import myfunctions, mess 

x = input("Please enter a number whose square you want:")
x_square = myfunctions.square(float(x))
print("The square of " + x + " = " + str(x_square))

s = input("Please enter a message to print:")
obj = mess(s)
obj.out()
