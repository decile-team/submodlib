#A dryrun of evaluate() and marginalGain() functions of FacilityLocation
#over dummy data
from submodlib_cpp import FacilityLocation
from submodlib.helper import create_kernel
data = [[1,2, 3], [3,4, 5], [4, 5,6]]
K = create_kernel(data, 'dense','euclidean')
print(K)
s = {1}
obj = FacilityLocation(3,"dense", K, 2, False, s)

X = {1}
print(obj.evaluate(X))

X = {1,2}
print(obj.evaluate(X))


X = {1}
print(obj.marginalGain(X,2))
