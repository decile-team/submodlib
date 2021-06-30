#include<unordered_set>
#include <pybind11/numpy.h>

typedef long long int ll;
float dot_prod(std::vector<float> &v1, std::vector<float> &v2);
float mag(std::vector<float> &v);
float cosine_similarity(std::vector<float> &v1, std::vector<float> &v2);
float euclidean_distance(std::vector<float> &v1, std::vector<float> &v2);
float euclidean_similarity(std::vector<float> &v1, std::vector<float> &v2);

struct datapoint_pair
{
	ll i1;
	ll i2;
	float val;
	datapoint_pair(ll a = 0, ll b = 0, float c = 0);
};

bool operator < (datapoint_pair lval, datapoint_pair rval);

//std::vector<std::vector<float>> create_kernel(std::vector<std::vector<float>> &X, std::string metric, ll num_neigh);
pybind11::array_t<float> create_kernel(pybind11::array_t<float> &X, std::string metric, ll num_neigh);
std::vector<std::vector<float>> create_kernel_NS(std::vector<std::vector<float>> &X_ground,std::vector<std::vector<float>> &X_master, std::string metric);
std::unordered_set<ll> set_intersection(std::unordered_set<ll> const &a, std::unordered_set<ll> const &b);
std::unordered_set<ll> set_union(std::unordered_set<ll> const &a, std::unordered_set<ll> const &b);