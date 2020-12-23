#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<cmath>
#include<set>
#include<map>
#include"sparse_utils.h"

SparseSim::SparseSim(std::vector<float> a_val, std::vector<ll> a_count, std::vector<ll> a_col) : arr_val(a_val), arr_count(a_count), arr_col(a_col), num_ind(a_count.size()-1) 
{
	v_col_ID.resize(num_ind);
	v_val_map.resize(num_ind);
	ll lower_r, upper_r;
	for (ll r = 0; r < num_ind; ++r)
	{
		lower_r = arr_count[r];
		upper_r = arr_count[r + 1]; //Since, non-zero values have been stored in a_val and a_count in a row by row fashion, we can identify the range of indicies in a_val and a_count
									//corrosponding to a particular row by using the arr_count. 

		//Storing non-zero values and columns corrosponding to r in efficient containers for optimal retrival (O(log(n)))
		for (ll i = lower_r; i < upper_r; ++i) //[arr_count[i], arr_count[i+1]) is the interval of indicies in arr_val and arr_col which corrospond to the ith row
		{
			v_col_ID[r].insert(arr_col[i]);
			v_val_map[r][arr_col[i]] = arr_val[i];
		}
	}

}

SparseSim::SparseSim():arr_val(std::vector<float>()), arr_count(std::vector<ll>()), arr_col(std::vector<ll>()), num_ind(0){}

float SparseSim::get_val(ll r, ll c)
{
	if (r >= num_ind || c >= num_ind || r < 0 || c < 0)
	{
		std::cerr << "ERROR: Incorrect row/column provided\n";
		return -2;
	}

	if (v_col_ID[r].find(c) == v_col_ID[r].end())
	{
		return 0;
	}
	else
	{
		return v_val_map[r][c];
	}
}

std::vector<float> SparseSim::get_row(ll r)
{
	std::vector<float>res(num_ind);
	for(int i=0; i<num_ind; ++i)
	{
		res[i]=get_val(r,i);
	}
	return res;
}

std::vector<float> SparseSim::get_col(ll c)
{
	std::vector<float>res(num_ind);
	for(int i=0; i<num_ind; ++i)
	{
		res[i]=get_val(i,c);
	}
	return res;
}

