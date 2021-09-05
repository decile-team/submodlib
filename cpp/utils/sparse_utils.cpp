  
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<cmath>
#include<set>
#include"sparse_utils.h"


/*//A proposed optimization
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Change in constructor: Remove the body, it contains a nested loop that fills information from csr matrix in set for efficien access.
SparseSim::SparseSim(std::vector<float> a_val, std::vector<ll> a_count, std::vector<ll> a_col) : arr_val(a_val), arr_count(a_count), arr_col(a_col), num_ind(a_count.size()-1), alwaysNonZero(false) // O(num_ind*num_neigh*log(num_neigh)) (One time operation){}

//Reason: This loop takes O(num_ind*num_neigh*log(num_neigh)) and thus might be acting as a bottleneck.
//I have realized that, we can still have the efficient log(num_neigh) access without this preprocessing loop. 

//Change in get_val(): Column indicies (for a particular row) in arr_col vector will be present in sorted order. Therefore, we can simply use binary binary search to
//check if corrosponding to a given row, given column has a non-zero value or not.
//Here we can use std::binary_search() and std::lower_bound() to do this as shown next. 
float SparseSim::get_val(ll r, ll c) // O(log(num_neigh))
{
	if (r >= num_ind || c >= num_ind || r < 0 || c < 0)
	{
		std::cerr << "ERROR: Incorrect row/column provided\n";
		return -2;
	}

	ll lower_i = arr_count[r];
	ll upper_i = arr_count[r + 1]; 
	
	if (std::binary_search(arr_col.begin()+lower_i, arr_col.begin()+upper_i, c))//if column c wrt row r has a non-zero value 
	{	
		//obtain non-zero value stored at (r,c) 
		auto it = std::lower_bound(arr_col.begin()+lower_i, arr_col.begin()+upper_i, c);
		return arr_val[*it];
	}
	else
	{
		return 0;
	}
}

*/

//TODO: overload = for deep copy

SparseSim::SparseSim(std::vector<float> const &a_val, std::vector<ll> const &a_count, std::vector<ll> const &a_col) : arr_val(a_val), arr_count(a_count), arr_col(a_col), num_ind(a_count.size()-1), alwaysNonZero(false) // O(num_ind*num_neigh*log(num_neigh)) (One time operation)
{
	v_col_ID.resize(num_ind);
	v_val_map.resize(num_ind);
	ll lower_i, upper_i;
	for (ll r = 0; r < num_ind; ++r)
	{
		//Since, non-zero values have been stored in arr_val and arr_count in a row by row fashion, we can identify the range of indicies in arr_val and arr_count
		//corrosponding to a particular row by using the arr_count as done below.
		lower_i = arr_count[r];
		upper_i = arr_count[r + 1]; 

		//In following loop, we are storing non-zero values and columns corrosponding to r in efficient containers for optimal retrival
		for (ll i = lower_i; i < upper_i; ++i) //[arr_count[i], arr_count[i+1]) is the interval of indicies in arr_val and arr_col which corrospond to the ith row
		{
			v_col_ID[r].insert(arr_col[i]);
			v_val_map[r][arr_col[i]] = arr_val[i];
		}
	}

}

SparseSim::SparseSim(std::vector<float> const &a_val, std::vector<ll> const &a_col, ll nn, ll num):arr_val(a_val), arr_col(a_col), num_neigh(nn), num_ind(num), alwaysNonZero(true) // O(num_ind*num_neigh*log(num_neigh)) (One time operation)
{
	v_col_ID.resize(num_ind);
	v_val_map.resize(num_ind);
	ll lower_i, upper_i;
	for (ll r = 0; r < num_ind; ++r)
	{
		//In this constructor, we assume that all similarity values are non-zero and thus arr_count will be a constant vector
		//whose elements can be inferred mathematically (as shown below) and thus there is no need to store it.
		lower_i = r*num_neigh;
		upper_i = (r + 1)*num_neigh; 

		
		for (ll i = lower_i; i < upper_i; ++i) 
		{
			v_col_ID[r].insert(arr_col[i]);
			v_val_map[r][arr_col[i]] = arr_val[i];
		}
	}
}

SparseSim::SparseSim():arr_val(std::vector<float>()), arr_count(std::vector<ll>()), arr_col(std::vector<ll>()), num_ind(0){}

float SparseSim::get_val(ll r, ll c) // O(log(num_neigh))
{
	if (r >= num_ind || c >= num_ind || r < 0 || c < 0)
	{
		std::cerr << "ERROR: Incorrect row/column provided\n";
		return -2;
	}
	if (v_col_ID[r].find(c) == v_col_ID[r].end())
	{
		// c is not in the neighbor list of r
		if (v_col_ID[c].find(r) == v_col_ID[c].end()) {
			// r is not in neighbor list of c
			return 0;
		} else {
			return v_val_map[c][r];
		}
	}
	else
	{
		return v_val_map[r][c];
	}
	
}

std::vector<float> SparseSim::get_row(ll r) // O(num_ind) (More optimal then get_col() in case of csr)
{
	std::vector<float>res(num_ind,0);
	if (r >= num_ind || r < 0)
	{
		std::cerr << "ERROR: Incorrect row provided\n";
		return std::vector<float>();
	}
	
	ll lower_i, upper_i;
	if(!alwaysNonZero)
	{
		lower_i = arr_count[r];
		upper_i = arr_count[r + 1]; 
	}
	else
	{
		lower_i = r*num_neigh;
		upper_i = (r + 1)*num_neigh; 	
	}
	
	for (ll i = lower_i; i < upper_i; ++i) 
	{
		res[arr_col[i]] = arr_val[i];
	}

	return res;
}

std::vector<float> SparseSim::get_col(ll c) // O(num_ind*log(num_neigh))
{
	std::vector<float>res(num_ind, 0);
	if (c >= num_ind || c < 0)
	{
		std::cerr << "ERROR: Incorrect column provided\n";
		return std::vector<float>();
	}
	for(int i=0; i<num_ind; ++i)
	{
		res[i]=get_val(i,c);
	}
	return res;
}
