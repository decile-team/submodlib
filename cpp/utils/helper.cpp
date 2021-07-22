/*
helper.cpp
Author: Vishal Kaushal
g++ -std=c++11 helper.cpp -o helper
./helper
Defines helper functions and compares performance of two different approaches of kernel computation in C++
*/

#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<unordered_map>
#include<algorithm>
#include <chrono>
#include <iomanip>
#include"helper.h"
#include <iostream>
#include <fstream>

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


std::vector<std::vector<float>> create_kernel(std::vector<std::vector<float>> &X, std::string metric, ll num_neigh)
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
			// s = metric == "euclidean" ? euclidean_similarity(X[r], X[c]) : cosine_similarity(X[r], X[c]);
			if (metric == "euclidean") {
				s = euclidean_similarity(X[r], X[c]);
			} else if (metric == "cosine") {
				s = cosine_similarity(X[r], X[c]);
			} else if (metric == "dot") {
				s = dot_prod(X[r], X[c]);
			} else {
				throw "Unsupported metric for kernel creation in CPP";
			}
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
			if (metric == "euclidean") {
				k_dense[r][c]=euclidean_similarity(X_master[r], X_ground[c]);
			} else if (metric == "cosine") {
				k_dense[r][c]=cosine_similarity(X_master[r], X_ground[c]);
			} else if (metric == "dot") {
				k_dense[r][c]=dot_prod(X_master[r], X_ground[c]);
			} else {
				throw "Unsupported metric for kernel computation in CPP";
			}
		}
	}
	return k_dense;
	
}

//Returns a Dense kernel of n_ground x n_ground
std::vector<std::vector<float>> create_square_kernel_dense(std::vector<std::vector<float>> &X_ground, std::string metric)
{
	ll n_ground = X_ground.size();
	std::vector<std::vector<float>>k_dense(n_ground, std::vector<float>(n_ground, 0)); 
	float sim = 0;
    float a_norm = 0, b_norm = 0;
	if (metric == "euclidean") {
		for (int r = 0; r < n_ground; ++r) {
			k_dense[r][r] = 1.0;
			for (int c = r + 1; c < n_ground; ++c) {
				sim = euclidean_similarity(X_ground[r], X_ground[c]);
				k_dense[r][c] = sim;
				k_dense[c][r] = sim;
			}
		}
	} else if (metric == "cosine") {
		for (int r = 0; r < n_ground; ++r) {
			a_norm = sqrt(dot_prod(X_ground[r], X_ground[r]));
			k_dense[r][r] = 1.0;
			for (int c = r + 1; c < n_ground; ++c) {
				sim = dot_prod(X_ground[r], X_ground[c]);
				b_norm = sqrt(dot_prod(X_ground[c], X_ground[c]));
				sim = (a_norm * b_norm) > 0 ? sim / (a_norm * b_norm) : 0;
				k_dense[r][c] = sim;
				k_dense[c][r] = sim;
			}
		}
    } else if (metric == "dot") {
		for (int r = 0; r < n_ground; ++r) {
			for (int c = r; c < n_ground; ++c) {
				sim = dot_prod(X_ground[r], X_ground[c]);
				k_dense[r][c] = sim;
				k_dense[c][r] = sim;
			}
		}
    } else {
		throw "Unsupported metric for kernel computation in CPP";
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

int main(int argc, char** argv) {
	std::cout << "Checking kernel values" << std::endl;
	std::vector<std::vector<float>> test_features {
                { 4.5, 13.5 },
                { 5, 13.5 },
                { 5.5, 13.5 }
            };
	std::vector<std::vector<float> > kernel1;
	std::vector<std::vector<float> > kernel2;
	std::cout << "Computing sample kernel using current method: \n";
	kernel1 = create_kernel(test_features, "euclidean", 3);
	std::cout << "[ ";
	for(auto row : kernel1) {
		std::cout << "[";
		for (auto column : row ) {
            std::cout << column << " ";
		}
		std::cout << "]\n";
	}
	std::cout << "]\n";
	std::cout << "Computing sample kernel using new method: \n";
	kernel2 = create_square_kernel_dense(test_features, "euclidean");
	std::cout << "[ ";
	for(auto row : kernel2) {
		std::cout << "[";
		for (auto column : row ) {
            std::cout << column << " ";
		}
		std::cout << "]\n";
	}
	std::cout << "]\n";
	int num_executions = 10;
	int num_places = 6;
	int num_features = 1024;
	std::vector<int> params{ 50, 100, 200, 500, 1000, 5000 };
	//std::vector<int> params{ 50, 100 };
	for (auto num_samples: params) {
		std::ofstream csvfile;
		csvfile.open ("cpp_kernel_creation_timings_" + std::to_string(num_samples) + "_.csv");
		csvfile << "Num Samples, Function, Time\n";
		std::cout << "Running for " << num_samples << " samples\n";
		std::vector<std::vector<float> > features;
		for(int i = 0; i < num_samples; i++) {
			std::vector<float> featuresOfOneElement;
			for(int j = 0; j < num_features; j++) {
				featuresOfOneElement.push_back((float)(rand()%10)/5.0);
			}
			features.push_back(featuresOfOneElement);
		}
		csvfile << num_samples << "," << "current,";
		std::cout << "Computing kernel using current method\n";
		double totalTime = 0;
		for(int i=0; i<num_executions; i++) {
			std::vector<std::vector<float> > kernel;
			auto start = std::chrono::high_resolution_clock::now();
			kernel = create_kernel(features, "euclidean", num_samples);
			auto stop = std::chrono::high_resolution_clock::now();
			double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
			duration *= 1e-9;
			std::cout << "Time taken in seconds: " << std::fixed << duration << std::setprecision(9) << std::endl;
			totalTime += duration;
			csvfile << duration << ",";
		}
		double averageTime = totalTime/num_executions;
		csvfile << averageTime << "\n";
		csvfile << num_samples << "," << "new,";
		std::cout << "Computing kernel using new method\n";
		totalTime = 0;
		for(int i=0; i<num_executions; i++) {
			std::vector<std::vector<float> > kernel;
			auto start = std::chrono::high_resolution_clock::now();
			kernel = create_square_kernel_dense(features, "euclidean");
			auto stop = std::chrono::high_resolution_clock::now();
			double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
			duration *= 1e-9;
			std::cout << "Time taken in seconds: " << std::fixed << duration << std::setprecision(9) << std::endl;
			totalTime += duration;
			csvfile << duration << ",";
		}
		averageTime = totalTime/num_executions;
		csvfile << averageTime << "\n";
		csvfile.close();
	}
}


