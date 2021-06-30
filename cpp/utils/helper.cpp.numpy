#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<unordered_map>
#include<algorithm>

#include"helper.h"

float dot_prod(std::vector<float> &v1, std::vector<float> &v2)
{
	float dp = 0;
	for (ll i = 0; i < v1.size(); ++i)//O(n) where n is number of features
	{
		dp += (v1[i] * v2[i]);
	}
	return dp;
}


float mag(std::vector<float> &v)
{
	float sum = 0;
	sum = dot_prod(v, v);
	return std::sqrt(sum);
}


float cosine_similarity(std::vector<float> &v1, std::vector<float> &v2)
{
	float dp, mag_v1 = mag(v1), mag_v2 = mag(v2), res;
	if(mag_v1==0 || mag_v2==0)//This is being done to keep results with sklearn (Otherwise, C++ would return NaN)
	{
		return 0; 
	}
	dp = dot_prod(v1, v2);
	res = dp / (mag_v1*mag_v2);
	return res;
}


float euclidean_distance(std::vector<float> &v1, std::vector<float> &v2)
{
	float sum = 0;
	for (ll i = 0; i < v1.size(); ++i)//Here we are not calling dot_prod() funct because for that we need to create a temp vector which can lead to increased space complexity
	{
		float diff = (v1[i] - v2[i]);
		sum += (diff*diff);
	}
	return std::sqrt(sum);

}

float euclidean_similarity(std::vector<float> &v1, std::vector<float> &v2)
{
	float ED = euclidean_distance(v1, v2);
	float gamma = 1.0 / v1.size();
	float ES = std::exp(-ED * gamma);
	return ES;
}



datapoint_pair::datapoint_pair(ll a, ll b, float c) :i1(a), i2(b), val(c) {}

bool operator < (datapoint_pair lval, datapoint_pair rval)
{
	return lval.val > rval.val;//because we want to create min heap
}

void update_heap(std::vector<std::vector<datapoint_pair>> &v_h, ll num_neigh, ll r, ll c, float s)
{
	if (v_h[r].size() < num_neigh)//populate heap till it has num_neigh elements
	{
		v_h[r].push_back(datapoint_pair(r, c, s));
		std::push_heap(v_h[r].begin(), v_h[r].end()); //Insert takes O(log(num_neigh))
	}
	else//once there are num_neigh elements in heap, there are 3 possibilities now
		//1) All biggest num_neigh similarities are already in heap
		//2) Some of biggest num_neigh similarities are in heap.
		//3) All biggest num_neigh similarities are not in heap
		//So on further traversal of similarities, we insert a similarity in the heap only if its heavier/greater than root of heap.
		//If that's the case, we delete the similarity at root of heap and insert the heavier/greater similarity in heap. Otherwise we skip the similarity.
	{
		if (v_h[r].front().val < s)
		{
			std::pop_heap(v_h[r].begin(), v_h[r].end());//smaller values are squeezed out of the heap 
			v_h[r][v_h[r].size()-1] = datapoint_pair(r, c, s);
			std::push_heap(v_h[r].begin(), v_h[r].end());//larger encountered values are inserted in the heap
		}
	}
}


/*std::vector<std::vector<float>> create_kernel(std::vector<std::vector<float>> &X, std::string metric, ll num_neigh)
//returns a similarity matrix where only num_neigh nearest neighbors are kept non-zero, rest are made zero
//It returns a vector of vector to pybind11 irrespective of the mode. Then pybind11 can forward it as list of list to Python FL
//which can use it to obtain matrix in desired mode.
{
	ll n = X.size();
	const int def_unvisited = -2, def_visited = 2;//default values (for purpose of memoization check)
	//std::vector<std::vector<float>>memo(n, std::vector<float>(n, def_unvisited));//memoization matrix
	//std::vector<std::vector<float>>sim(n, std::vector<float>(n, 0));//result matrix

	//Here, I have used the a min heap (not max heap) to mantain k highest similarities. This heap will be mantained such that
	//smallest similarity (among k highest similarities ) is at the root and all larger similarity successors of this root.
	std::vector<std::vector<datapoint_pair>>v_h(n); //vector of min heaps (here ith element is a min heap containing num_neigh nearest neighbors of ith example)
	std::vector<std::vector<float>>content(3); //This vector will contain 3 vectors containing non-zero values, their row id and their column id respt.
	float s;
	ll count = 0;

	//Upper triangular traversal
	for (ll r = 0; r < n; ++r)
	{
		for (ll c = r; c < n; ++c)
		{
			s = metric == "euclidean" ? euclidean_similarity(X[r], X[c]) : cosine_similarity(X[r], X[c]);
			
			update_heap(v_h, num_neigh, r, c, s);
			if(r!=c)
			{
				update_heap(v_h, num_neigh, c, r, s);
			}

		}
	}

	for (ll r = 0; r < v_h.size(); ++r)//build a matrix where for rth row only num_neigh nearest neighbors are assigned non-zero similarities
	{
		//std::cout<<r<<" "<<v_h[r].size()<<"\n";
		while (v_h[r].size() != 0)//place nearest neighbors of rth datapoint in content vector 
		{
			
			datapoint_pair obj = v_h[r].front();
			//sim[obj.i1][obj.i2] = obj.val;
			//std::cout<<obj.i1<<" "<<obj.i2<<" "<<obj.val<<"\n";
			content[0].push_back(obj.val);//non-zero values
			content[1].push_back(obj.i1);//row id
			content[2].push_back(obj.i2);//column id

			std::pop_heap(v_h[r].begin(), v_h[r].end());
			v_h[r].erase(v_h[r].begin() + (v_h[r].size() - 1)); //TODO: Modify this part to avoid use of erase()
		}
	}

	//return sim;
	return content;
}*/

pybind11::array_t<float> create_kernel(pybind11::array_t<float> &X, std::string metric, ll num_neigh)
//returns a similarity matrix where only num_neigh nearest neighbors are kept non-zero, rest are made zero
{
	pybind11::buffer_info bufX = X.request();
	ll n = bufX.shape[0];
	float *ptr = (float *)bufX.ptr;
	size_t width = bufX.shape[1];
	const int def_unvisited = -2, def_visited = 2;//default values (for purpose of memoization check)
	//std::vector<std::vector<float>>memo(n, std::vector<float>(n, def_unvisited));//memoization matrix
	//std::vector<std::vector<float>>sim(n, std::vector<float>(n, 0));//result matrix

	//Here, I have used the a min heap (not max heap) to mantain k highest similarities. This heap will be mantained such that
	//smallest similarity (among k highest similarities ) is at the root and all larger similarity successors of this root.
	std::vector<std::vector<datapoint_pair>>v_h(n); //vector of min heaps (here ith element is a min heap containing num_neigh nearest neighbors of ith example)
	std::vector<std::vector<float>>content(3); //This vector will contain 3 vectors containing non-zero values, their row id and their column id respt.

	float s;
    
	if(metric == "cosine") {
		//Upper triangular traversal
		for (ll r = 0; r < n; ++r)
		{
			for (ll c = r; c < n; ++c)
			{
                float dp_rr = 0;
				for (size_t idx = 0; idx < width; idx++) {
					int curIdr = r*width + idx;
					dp_rr += (ptr[curIdr] * ptr[curIdr]);
				}
				float mag_vr = std::sqrt(dp_rr);

				float dp_cc = 0;
				for (size_t idx = 0; idx < width; idx++) {
					int curIdc = c*width + idx;
					dp_cc += (ptr[curIdc] * ptr[curIdc]);
				}
				float mag_vc = std::sqrt(dp_cc);

				if(mag_vr==0 || mag_vc==0)//This is being done to keep results with sklearn (Otherwise, C++ would return NaN)
				{
					s=0;
				} else {
					float dp_rc = 0;
					for (size_t idx = 0; idx < width; idx++) {
						int curIdr = r*width + idx;
						int curIdc = c*width + idx;
						dp_rc += (ptr[curIdr] * ptr[curIdc]);
					}
					s = dp_rc / (mag_vr*mag_vc);
				}
				
				update_heap(v_h, num_neigh, r, c, s);
				if(r!=c)
				{
					update_heap(v_h, num_neigh, c, r, s);
				}
			}
		}
	} else {
		//Upper triangular traversal
		for (ll r = 0; r < n; ++r)
		{
			for (ll c = r; c < n; ++c)
			{
				float sum = 0;
				for (size_t idx = 0; idx < width; idx++) {
					int curIdr = r*width + idx;
					int curIdc = c*width + idx;
					float diff = ptr[curIdr] - ptr[curIdc];
					sum += (diff*diff);
				}
				float ED = std::sqrt(sum);
				float gamma = 1.0 / width;
				s = std::exp(-ED * gamma);		
				update_heap(v_h, num_neigh, r, c, s);
				if(r!=c)
				{
					update_heap(v_h, num_neigh, c, r, s);
				}
			}
		}
	}
	ll count = 0;

	for (ll r = 0; r < v_h.size(); ++r)//build a matrix where for rth row only num_neigh nearest neighbors are assigned non-zero similarities
	{
		//std::cout<<r<<" "<<v_h[r].size()<<"\n";
		while (v_h[r].size() != 0)//place nearest neighbors of rth datapoint in content vector 
		{
			
			datapoint_pair obj = v_h[r].front();
			//sim[obj.i1][obj.i2] = obj.val;
			//std::cout<<obj.i1<<" "<<obj.i2<<" "<<obj.val<<"\n";
			content[0].push_back(obj.val);//non-zero values
			content[1].push_back(obj.i1);//row id
			content[2].push_back(obj.i2);//column id
			count++;

			std::pop_heap(v_h[r].begin(), v_h[r].end());
			v_h[r].erase(v_h[r].begin() + (v_h[r].size() - 1)); //TODO: Modify this part to avoid use of erase()
		}
	}
	// allocate the output buffer

	pybind11::array_t<float> result = pybind11::array_t<double>(count*3);
	pybind11::buffer_info bufRes = result.request(); // acquire buffer info
	float *ptrRes = (float *)bufRes.ptr;
	int yIndx = 0;
	for(auto vec: content) {
		int xIndx = 0;
        for(auto elem: vec) {
            int curIdx = yIndx*width + xIndx;
            ptrRes[curIdx] = elem;
			xIndx++;
		}
		yIndx++;
	}
	/* Reshape result to have same shape as input */
    result.resize({3 , (int)count});
	//return sim;
	//return content;
	return result;
}

//Returns a Dense of n_master x n_ground
std::vector<std::vector<float>> create_kernel_NS(std::vector<std::vector<float>> &X_ground,std::vector<std::vector<float>> &X_master, std::string metric)
{
	ll n_ground = X_ground.size();
	ll n_master = X_master.size();
 
	std::vector<std::vector<float>>k_dense(n_master, std::vector<float>(n_ground)); 
	float s;
	ll count = 0;

	for(int r=0;r<X_master.size();++r)
	{
		for(int c=0;c<X_ground.size();++c)
		{
			k_dense[r][c]= metric == "euclidean" ? euclidean_similarity(X_master[r], X_ground[c]) : cosine_similarity(X_master[r], X_ground[c]);
		}
	}

	return k_dense;
	
}

std::unordered_set<ll> set_intersection(std::unordered_set<ll> const &a, std::unordered_set<ll> const &b) {
	std::unordered_set<ll> c;
	//looping over the elements of the first set, hence better when first set is smaller
	for (auto element : a) {
		if (b.count(element) > 0) {
			c.insert(element);
		}
	}
	return c;
}

std::unordered_set<ll> set_union(std::unordered_set<ll> const &a, std::unordered_set<ll> const &b) {
	std::unordered_set<ll> c;
	for (auto elem: b){
			c.insert(elem);
	}
	//looping over the elements of the first set, hence better when first set is smaller
	for (auto element : a) {
		c.insert(element);
	}
	return c;
}


