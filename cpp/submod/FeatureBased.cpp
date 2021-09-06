#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"FeatureBased.h"

FeatureBased::FeatureBased(ll n_, Type type_, std::vector<std::vector<std::pair<int, float>>> const &groundSetFeatures_, int numFeatures_, std::vector<float> const& featureWeights_): n(n_), type(type_), groundSetFeatures(groundSetFeatures_), numFeatures(numFeatures_), featureWeights(featureWeights_)  {
	//Moved these error checks to Python
	// if(numFeatures != featureWeights.size()) {
	// 	throw "Mismatch in numFeatures and featureWeights.size() or in ";
	// }
	// if(groundSetFeatures.size() != n) {
	// 	throw "Mismatch in n and groundSetFeatures.size()";
	// }

	sumOfFeaturesAcrossX.resize(numFeatures, 0);
}

// FeatureBased* FeatureBased::clone() {
// 	return NULL;
// }


double FeatureBased::transform(double val) {
	//std::cout << "Type = " << type << "\n";
	switch(type) {
		case inverse:
		    //std::cout << "Inverse\n";
		    return (1-1/(val+1));
		case logarithmic:
			//std::cout << "Logarithmic\n";
		    return log(1 + val);
		case squareRoot:
		    //std::cout << "Sqrt\n";
		    return sqrt(val);
	}
}

double FeatureBased::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}

	// std::cout << "Received subset = {";
	// for(auto elem: X) {
	// 	std::cout << elem << ", ";
	// }
	// std::cout << "\n";
	std::vector<double> featuresSum (numFeatures, 0);
    for(auto elem: X) {
		for (int j = 0; j < groundSetFeatures[elem].size(); j++){ 
			featuresSum[groundSetFeatures[elem][j].first] += groundSetFeatures[elem][j].second;
		}
	}
	// std::cout << "Features sum = {";
	// for(auto elem: featuresSum) {
	// 	std::cout << elem << ", ";
	// }
	// std::cout << "\n";
	for(int i=0; i<numFeatures; i++) {
		//result += featureWeights[i]*transform(featuresSum[i]);
		// double temp = transform(featuresSum[i]);
		// double increase = featureWeights[i]*temp;
		// std::cout << "Val: " << featuresSum[i] << " Transform: " << temp << " weight: " << featureWeights[i] << " Increment: " << increase;
		// result += increase;
		// std::cout <<" New result: " << result << "\n";
		result += featureWeights[i]*transform(featuresSum[i]);
	}
	return result;
}

double FeatureBased::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	for (int i=0; i<numFeatures; i++){
		result += featureWeights[i]*transform(sumOfFeaturesAcrossX[i]);
	}
	return result;
}


double FeatureBased::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	std::vector<double> featuresSum (numFeatures, 0);
	for(auto elem: X) {
		for (int j = 0; j < groundSetFeatures[elem].size(); j++){ 
			featuresSum[groundSetFeatures[elem][j].first] += groundSetFeatures[elem][j].second;
		}
	}
	double temp;
	double diff;
	for (int i=0; i<groundSetFeatures[item].size(); i++){
		temp = featuresSum[groundSetFeatures[item][i].first];
		diff = transform(temp + groundSetFeatures[item][i].second) - transform(temp);
		gain += featureWeights[groundSetFeatures[item][i].first] * diff;
	}

	return gain;
}


double FeatureBased::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	double temp;
	double diff;
	for (int i=0; i<groundSetFeatures[item].size(); i++){
		temp = sumOfFeaturesAcrossX[groundSetFeatures[item][i].first];
		diff = transform(temp + groundSetFeatures[item][i].second) - transform(temp);
		gain += featureWeights[groundSetFeatures[item][i].first] * diff;
	}

	return gain;
}

void FeatureBased::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	for (int i=0; i<groundSetFeatures[item].size(); i++){
		sumOfFeaturesAcrossX[groundSetFeatures[item][i].first] += groundSetFeatures[item][i].second;
	}
}

std::unordered_set<ll> FeatureBased::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> FeatureBased::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	if(optimizer == "NaiveGreedy") {
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "LazyGreedy") {
//         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else if(optimizer == "LazierThanLazyGreedy") {
//         return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else {
// 		throw "Error: Invalid Optimizer";
// 	}
// }

void FeatureBased::clearMemoization() {
	for(int i=0;i<numFeatures;++i) {
		sumOfFeaturesAcrossX[i]=0;
	}
}

void FeatureBased::setMemoization(std::unordered_set<ll> const &X) 
{
    clearMemoization();
    std::unordered_set<ll> temp;
    for (auto elem: X)
	{	
		updateMemoization(temp, elem);
		temp.insert(elem);	
	}
}


