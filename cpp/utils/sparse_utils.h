#ifndef SPARSEUTILS_H
#define SPARSEUTILS_H
#include<map>
#include<set>
typedef long long int ll;

class SparseSim 
{
	std::vector<float>arr_val; //contains non - zero values in matrix(row major traversal)
	std::vector<ll>arr_count; //contains cumulitive count of non-zero elements upto but not including current row
	std::vector<ll>arr_col; //contains col index corrosponding to non-zero values in arr_val
	ll num_ind;//num of rows/cols in the similarity matrix
	ll num_neigh;//num of nearest neighbors being considered
	bool alwaysNonZero;//This flag tells if similarity values will be always (exactly) non Zero 
					   //or if zero values are also possible (say in case of orthogonal datapoints)
	std::vector<std::set<ll>> v_col_ID;
	std::vector<std::map<ll, float>> v_val_map;

public:
	SparseSim(std::vector<float> const &a_val, std::vector<ll> const &a_count, std::vector<ll> const &a_col);//Use this signature if (exact) 0 similarity is possible
	SparseSim(std::vector<float> const &a_val, std::vector<ll> const &a_col, ll nn, ll num);//Use this signature if (exact) zero similarity is impossible
	SparseSim();
	float get_val(ll r, ll c);
	std::vector<float> get_row(ll r);
	std::vector<float> get_col(ll c);
};
#endif