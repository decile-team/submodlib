#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<unordered_map>
#include<algorithm>

#include"helper.h"

//typedef long long int ll;

float dot_prod(std::vector<float>v1, std::vector<float>v2)
{
	float dp = 0;
	for (ll i = 0; i < v1.size(); ++i)//O(n) where n is number of features
	{
		dp += (v1[i] * v2[i]);
	}
	return dp;
}


float mag(std::vector<float>v)
{
	float sum = 0;
	sum = dot_prod(v, v);
	return std::sqrt(sum);
}


float cosine_similarity(std::vector<float>v1, std::vector<float>v2)
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


float euclidean_distance(std::vector<float>v1, std::vector<float>v2)
{
	float sum = 0;
	for (ll i = 0; i < v1.size(); ++i)//Here we are not calling dot_prod() funct because for that we need to create a temp vector which can lead to increased space complexity
	{
		float diff = (v1[i] - v2[i]);
		sum += (diff*diff);
	}
	return std::sqrt(sum);

}

float euclidean_similarity(std::vector<float>v1, std::vector<float>v2)
{
	float ED = euclidean_distance(v1, v2);
	float gamma = 1.0 / v1.size();
	float ES = std::exp(-ED * gamma);
	return ES;
}



datapoint_pair::datapoint_pair(ll a = 0, ll b = 0, float c = 0) :i1(a), i2(b), val(c) {}

bool operator < (datapoint_pair lval, datapoint_pair rval)
{
	return lval.val > rval.val;//because we want to create min heap
}


std::vector<std::vector<float>> create_kernel(std::vector<std::vector<float>>X, std::string metric, ll num_neigh)
//returns a similarity matrix where only num_neigh nearest neighbors are kept non-zero, rest are made zero
//It returns a vector of vector to pybind11 irrespective of the mode. Then pybind11 can forward it as list of list to Python FL
//which can use it to obtain matrix in desired mode.
{
	ll n = X.size();
	const int def_unvisited = -2, def_visited = 2;//default values (for purpose of memoization check)
	std::vector<std::vector<float>>memo(n, std::vector<float>(n, def_unvisited));//memoization matrix
	std::vector<std::vector<float>>sim(n, std::vector<float>(n, 0));//result matrix

	//Here, I have used the a min heap (not max heap) to mantain k highest similarities. This heap will be mantained such that
	//smallest similarity (among k highest similarities ) is at the root and all larger similarity successors of this root.
	std::vector<std::vector<datapoint_pair>>v_h(n); //vector of min heaps (here ith element is a min heap containing num_neigh nearest neighbors of ith example)
	float s;
	ll count = 0;
	for (ll r = 0; r < n; ++r)
	{
		for (ll c = 0; c < n; ++c)
		{
			if (memo[r][c] == def_unvisited)//if similarity for given vectors hasn't been previously computed only then compute
			{
				s = metric == "euclidean" ? euclidean_similarity(X[r], X[c]) : cosine_similarity(X[r], X[c]);

				memo[r][c] = s;
				memo[c][r] = s;

			}
			else//if similarity for given vectors has been previously computed, simply reuse it
			{
				s = memo[r][c];
			}
			if (v_h[r].size() < num_neigh)//populate heap till it has num_neigh elements
			{
				//std::cout<<"within: "<< r<<" "<<c<<" "<<" "<<s<<"\n";
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
				//std::cout<<v_h[r].front().val<<" "<<s<<"\n";
				if (v_h[r].front().val < s)
				{
					//std::cout<<"without: "<<v_h[r].front().val<<" "<<r<<" "<<c<<" "<<" "<<s<<"\n";
					std::pop_heap(v_h[r].begin(), v_h[r].end());//smaller values are squeezed out of the heap 
					v_h[r][v_h[r].size()-1] = datapoint_pair(r, c, s);
					std::push_heap(v_h[r].begin(), v_h[r].end());//larger encountered values are inserted in the heap
				}
			}
		}
	}

	for (ll r = 0; r < v_h.size(); ++r)//build a matrix where for rth row only num_neigh nearest neighbors are assigned non-zero similarities
	{
		//std::cout<<r<<" "<<v_h[r].size()<<"\n";
		while (v_h[r].size() != 0)//place nearest neighbors of rth datapoint in similarity matrix 
		{
			
			datapoint_pair obj = v_h[r].front();
			sim[obj.i1][obj.i2] = obj.val;
			//std::cout<<obj.i1<<" "<<obj.i2<<" "<<obj.val<<"\n";
			std::pop_heap(v_h[r].begin(), v_h[r].end());
			v_h[r].erase(v_h[r].begin() + (v_h[r].size() - 1)); //TODO: Modify this part to avoid use of erase()
		}
	}

	return sim;
}

