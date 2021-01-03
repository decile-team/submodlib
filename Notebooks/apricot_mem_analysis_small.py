from apricot import FacilityLocationSelection
from memory_profiler import profile
import numpy as np
from scipy import sparse
import scipy

#data = None
#with open('/content/drive/MyDrive/submodlib_data/large_data.npy', 'rb') as f:
 #       data = np.load(f, allow_pickle=True)

data =np.array( [(4.5,13.5), (5,13.5), (5.5,13.5), (14.5,13.5), (15,13.5), (15.5,13.5),
(4.5,13), (5,13), (5.5,13), (14.5,13), (15,13), (15.5,13),
(4.5,12.5), (5,12.5), (5.5,12.5), (14.5,12.5), (15,12.5), (15.5,12.5),
(4.5,7.5), (5,7.5), (5.5,7.5), (14.5,7.5), (15,7.5), (15.5,7.5),
(4.5,7), (5,7), (5.5,7), (14.5,7), (15,7), (15.5,7),
(4.5,6.5), (5,6.5), (5.5,6.5), (14.5,6.5), (15,6.5), (15.5,6.5),
(7.5,10), (12.5,10), (10,12.5), (10,7.5), (4.5, 15.5), (5,9.5), (5,10.5)] )

@profile
def f_analysis():
  num_subsets = 10
  obj = FacilityLocationSelection(num_subsets, metric='euclidean', optimizer='lazy')
  obj=obj.fit(data)
  subsets= obj.transform(data)
    
    
if __name__ == '__main__':
    f_analysis()