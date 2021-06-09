# featureBased.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FeatureBased
from sklearn.preprocessing import MinMaxScaler

class FeatureBasedFunction(SetFunction):
	"""Implementation of the Feature-Based (FB) function.

	Feature based functions are essentially sums of concave over modular functions defined as:
	
	.. math::
			f(X) = \\sum_{f \\in F} w_f g(m_f(X))
	
	where :math:`g` is a concave function, :math:`{m_f}` are a set of feature scores, and :math:`f \\in F` are features. In case of images, features could be, for example, the features extracted from the second last layer of a ConvNet. 

	Feature Based functions model the notion of coverage over features.

	Parameters
	----------
	n : int
		Number of elements in the ground set. Must be > 0.
	
	features : list
		Feature vectors for the elements in the ground set. List of size n.
	
	numFeatures : int
		Dimensionality of the feature vectors of each ground set element.

	sparse : bool
		Indicates whether *features* contain sparse feature vectors. If True, *features* is expected to be a list of list of tuples where each sparse feature vector is represented by a list of tuples (i, j), i being the index of the non-ero feature value and j being the feature value. If False, the supplied *features* are converted into sparse representation internally.
	
	featureWeights : list
		Weights of features. List of size numFeatures.

	mode : FeatureBased.Type, optional
		The concave function to be used. Can be FeatureBased.logarithmic, FeatureBased.squareRoot, FeatureBased.inverse. Default is FeatureBased.logarithmic.
	
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

	