typedef long long int ll;

class SparseSim 
{
	std::vector<float>arr_val; //contains non - zero values in matrix(row major traversal)
	std::vector<ll>arr_count; //contains cumulitive count of non-zero elements upto but not including current row
	std::vector<ll>arr_col; //contains col index corrosponding to non-zero values in arr_val
	ll num_ind;//num of rows/cols in the similarity matrix

	std::vector<std::set<ll>>v_col_ID;
	std::vector<std::map<ll, float>>v_val_map;

public:
	SparseSim(std::vector<float> a_val, std::vector<ll> a_count, std::vector<ll> a_col);
	float get_val(ll r, ll c);
	std::vector<float> get_row(ll r);
	std::vector<float> get_col(ll c);
};