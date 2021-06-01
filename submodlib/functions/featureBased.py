# featureBased.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FeatureBased
from sklearn.preprocessing import MinMaxScaler

class FeatureBasedFunction(SetFunction):
	"""Implementation of the Feature-Based function.

	Feature based functions are essentially sums of concave over modular functions defined as
	
	.. math::
			f(X) = \\sum_{f \\in F} w_f g(m_f(X))
	
	where :math:`g` is a concave function, :math:`{m_f}_{f \\in F}` are a set of feature scores, and :math:`f \\in F` are features.

	Parameters
	----------
	n : int
		Number of elements in the ground set

	type : str
		Concave function to be used - sqrt, log, min, pow
	
	features : list
		Feature vectors for the elements in the ground set
	
	weights : list
		Weights of features
	
	thresh : float
		Threshold to be used for min function
	
	pow : int
		Exponent to be used for power function
	
	"""

	def __init__(self, n, features, numFeatures, sparse, featureWeights=None, mode=FeatureBased.logarithmic):
		self.n = n
		self.mode = mode
		self.features = features
		self.numFeatures = numFeatures
		self.featureWeights = featureWeights
		self.cpp_obj = None
		self.cpp_features = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in [FeatureBased.squareRoot, FeatureBased.inverse, FeatureBased.logarithmic]:
			raise Exception("ERROR: Incorrect mode. Must be one of 'squareRoot', 'inverse' or 'logarithmic'")

		if n != len(features):
			raise Exception("ERROR: Mismtach between n and len(features)")
		
		if (type(featureWeights) != type(None)):
			if numFeatures != len(featureWeights):
			    raise Exception("ERROR: Mismtach between numFeatures and len(featureWeights)")
		else:
			self.featureWeights = [1] * numFeatures

		#print("Features before normalization: ", features)
		
		#min-max normalize the features so that they are between 0 and 1
		featuresArray = np.array(features)
		norm = MinMaxScaler().fit(featuresArray)
		normalizedFeatures = norm.transform(featuresArray)
		features = normalizedFeatures.tolist()
		#print("Features after normalization: ", features)
		#convert the features into sparse representation (list of tuples) if not already
		self.cpp_features = []
		if not sparse:
			for i in range(len(features)):
				featureVec = []
				for j in range(len(features[i])):
					if (features[i][j] != 0):
						featureVec.append((j, features[i][j]))
				self.cpp_features.append(featureVec)
		else:
			self.cpp_features = features

		#print("Sparse representation:", self.cpp_features)
			
		self.cpp_obj = FeatureBased(self.n, self.mode, self.cpp_features, self.numFeatures, self.featureWeights)

		self.effective_ground = set(range(n))

	