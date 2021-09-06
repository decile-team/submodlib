#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "LazierThanLazyGreedyOptimizer.h"

struct classcomp {
  bool operator() (const std::pair<double, ll>& lhs, const std::pair<double, ll>& rhs) const
  {
    return ((lhs.first==rhs.first)?(lhs.second>rhs.second):(lhs.first > rhs.first));
  }
};

LazierThanLazyGreedyOptimizer::LazierThanLazyGreedyOptimizer() {}

bool LazierThanLazyGreedyOptimizer::equals(double val1, double val2,
                                           double eps) {
    if (abs(val1 - val2) < eps)
        return true;
    else {
        return false;
    }
}

// bool printSortedSet(std::set<std::pair<float, ll>> &sortedSet) {
//     std::cout << "[";
//     for (auto rev_it = sortedSet.rbegin(); rev_it != sortedSet.rend();
//          rev_it++) {
//         std::cout << "(" << (*rev_it).first << ", " << (*rev_it).second
//                   << "), ";
//     }
//     std::cout << "]\n";
// }

void printSortedSet(std::set<std::pair<double, ll>, classcomp> &sortedSet) {
    std::cout << "[";
    for (auto it = sortedSet.begin(); it != sortedSet.end();
         it++) {
        std::cout << "(" << (*it).first << ", " << (*it).second
                  << "), ";
    }
    std::cout << "]\n";
}

/*std::vector<std::pair<ll, float>>
LazierThanLazyGreedyOptimizer::maximize(SetFunction &f_obj, ll budget, bool
stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool
verbose=false) {
        //TODO: take care of handling equal guys later
        //TODO: take care of different sizes of each items - becomes a candidate
only if best and within budget, cost sensitive selection
        std::vector<std::pair<ll, float>>greedyVector;
        greedyVector.reserve(budget);
        std::unordered_set<ll> greedySet;
        greedySet.reserve(budget);
        ll rem_budget = budget;
        std::unordered_set<ll> remainingSet = f_obj.getEffectiveGroundSet();
        ll n = remainingSet.size();
        ll randomSetSize = ((float)n/budget)* log(1/epsilon);

        if (verbose) {
                std::cout << "Epsilon = " << epsilon << "\n";
                std::cout << "Random set size = " << randomSetSize << "\n";
                std::cout << "Ground set:" << std::endl;
                for(int i: remainingSet) {
                        std::cout << i << " ";
                }
                std::cout << "\n";
                std::cout << "Num elements in groundset = " <<
remainingSet.size() << std::endl; std::cout<<"Starting the LazierThanLazy greedy
algorithm\n"; std::cout << "Initial greedy set:" << std::endl; for(int i:
greedySet) { std::cout << i << " ";
                }
                std::cout << "\n";
        }
        f_obj.clearMemoization();
        ll best_id;
        float best_val;

  // initialize sorted list of initial gains for each element
        std::set<std::pair<float, ll>> sortedGains;
        // for each element in the ground set
        for (auto elem : remainingSet) {
                        sortedGains.insert(std::pair<float, ll>(
                                        f_obj.marginalGainWithMemoization(greedySet,
elem), elem));
        }

        if(verbose) {
                std::cout << "Initial sorted set = ";
                printSortedSet(sortedGains);
        }
  int i = 0;
        while (rem_budget > 0) {
                std::unordered_set<ll> randomSet;
                while(randomSet.size() < randomSetSize) {
                    ll elem = rand() % n; //TODO:serious issue, works only till
RAND_MAX
                                //std::cout << "Trying random element " << elem
<< "\n"; if((remainingSet.find(elem) != remainingSet.end()) &&
(randomSet.find(elem) == randomSet.end())){
            //valid element
                                                //std::cout << "Valid, adding
it\n"; randomSet.insert(elem);
                                }
                }
                if(verbose) {
                        std::cout << "Iteration " << i << "\n";
                        std::cout << "Random set = [";
                        for(auto elem: randomSet) {
                                std::cout << elem << " ";
                        }
                        std::cout << "\n";
                }
                if(verbose) std::cout << "Now running lazy greedy on the random
set\n"; ll best_id; float best_val; bool done = false; while (!done) { float
candidate_val; ll candidate_id; float newCandidateBound;
                        //std::cout << "Current sortedGains = ";
                        //printSortedSet(sortedGains);
                        for (auto rev_it = sortedGains.rbegin(); rev_it !=
sortedGains.rend(); rev_it++) {
                                //std::cout << "Checking " << (*rev_it).second
<< "...\n"; if (randomSet.find((*rev_it).second) != randomSet.end()) {
                                        //std::cout << "...present in random
set....\n"; candidate_id = (*rev_it).second; candidate_val = (*rev_it).first;
                                        newCandidateBound =
                                                                                f_obj.marginalGainWithMemoization(greedySet,
                                                                                                                                                                                                                        candidate_id);
                                        //std::cout << "Updated gain as per
updated greedy set = " << newCandidateBound << "\n"; auto nextElem =
std::next(rev_it, 1); if (newCandidateBound > (*nextElem).first) {
                                                //std::cout << "..better than
next best upper bound, selecting...\n"; best_id = candidate_id; best_val =
newCandidateBound; done = true; break; } else {
                                                //std::cout << "... NOT better
than next best upper bound, updating...\n"; done = false; break;
                                        }
                                } else {
                                        continue;
                                }
                        }
                        if(done) {
                                  //std::cout << "...removing from sorted
set\n"; sortedGains.erase(std::pair<float, ll>(candidate_val, candidate_id)); }
else {
                                  //std::cout << "...updating its value in
sorted set\n"; sortedGains.erase(std::pair<float, ll>(candidate_val,
candidate_id)); sortedGains.insert(std::pair<float, ll>(newCandidateBound,
candidate_id));
                        }
                }
                if(verbose) {
                        std::cout << "Next best item to add is " << best_id << "
and its value addition is " << best_val << "\n";
    }
                if ( (best_val < 0 && stopIfNegativeGain) || (equals(best_val,
0, 1e-5) && stopIfZeroGain) ) { break; } else {
                        f_obj.updateMemoization(greedySet, best_id);
                        greedySet.insert(best_id); //greedily insert the best
datapoint index of current iteration of while loop
                        greedyVector.push_back(std::pair<ll, float>(best_id,
best_val)); rem_budget-=1; remainingSet.erase(best_id); if(verbose) {
                                std::cout<<"Added element "<< best_id << " and
the gain is " << best_val <<"\n"; std::cout << "Updated greedySet: "; for(int i:
greedySet) { std::cout << i << " ";
                        }
                        std::cout << "\n";
                        }
                }
                i += 1;
        }
        return greedyVector;
}*/

std::vector<std::pair<ll, double>> LazierThanLazyGreedyOptimizer::maximize(
    SetFunction &f_obj, float budget, bool stopIfZeroGain,
    bool stopIfNegativeGain, float epsilon,
    bool verbose, bool showProgress, const std::vector<float>& costs, bool costSensitiveGreedy) {
    // TODO: take care of handling equal guys later
    // TODO: take care of different sizes of each items - becomes a candidate
    // only if best and within budget, cost sensitive selection
    std::vector<std::pair<ll, double>> greedyVector;
    std::unordered_set<ll> greedySet;
    if(costs.size()==0) {
		//every element is of same size, budget corresponds to cardinality
        greedyVector.reserve(budget);
	    greedySet.reserve(budget);
	}
    float rem_budget = budget;
    std::unordered_set<ll> remainingSet = f_obj.getEffectiveGroundSet();
    ll n = remainingSet.size();
    ll randomSetSize = ((double)n / budget) * log(1 / epsilon);

    if (verbose) {
        std::cout << "Epsilon = " << epsilon << "\n";
        std::cout << "Random set size = " << randomSetSize << "\n";
        std::cout << "Ground set:" << std::endl;
        for (int i : remainingSet) {
            std::cout << i << " ";
        }
        std::cout << "\n";
        std::cout << "Num elements in groundset = " << remainingSet.size()
                  << std::endl;
        std::cout << "Starting the LazierThanLazy greedy algorithm\n";
        std::cout << "Initial greedy set:" << std::endl;
        for (int i : greedySet) {
            std::cout << i << " ";
        }
        std::cout << "\n";
    }
    f_obj.clearMemoization();
    srand(1);
    ll best_id;
    double best_val;

    // initialize sorted list of initial gains for each element
    std::set<std::pair<double, ll>, classcomp> sortedGains;
    // for each element in the ground set
    for (auto elem : remainingSet) {
        sortedGains.insert(std::pair<double, ll>(
            f_obj.marginalGainWithMemoization(greedySet, elem, false), elem));
    }

    if (verbose) {
        std::cout << "Initial sorted set = ";
        printSortedSet(sortedGains);
    }
    int i = 0;
    int step = 1;
	int displayNext = step;
	int percent = 0;
    float N = rem_budget;
    int iter = 0;
    while (rem_budget > 0) {
        std::unordered_set<ll> randomSet;
        while (randomSet.size() < randomSetSize) {
            ll elem =
                rand() %
                n;  // TODO:serious issue, works only till RAND_MAX
                    // std::cout << "Trying random element " << elem << "\n";
            if ((remainingSet.find(elem) != remainingSet.end()) &&
                (randomSet.find(elem) == randomSet.end())) {
                // valid element
                // std::cout << "Valid, adding it\n";
                randomSet.insert(elem);
            }
        }
        if (verbose) {
            std::cout << "Iteration " << i << "\n";
            std::cout << "Random set = [";
            for (auto elem : randomSet) {
                std::cout << elem << " ";
            }
            std::cout << "\n";
            std::cout << "Now running lazy greedy on the random set\n";
        }
        ll best_id;
        double best_val;
        double candidate_val;
        ll candidate_id;
        double newCandidateBound;
        
        for (auto it = sortedGains.begin(); it != sortedGains.end();) {
            if (verbose) {
                std::cout << "Current sortedGains = ";
                printSortedSet(sortedGains);
                std::cout << "Checking what iterator is pointing at " << (*it).second << "...\n";
            }

            if (randomSet.find((*it).second) != randomSet.end()) {
                if (verbose) std::cout << "...present in random set....\n";
                candidate_id = (*it).second;
                candidate_val = (*it).first;
                newCandidateBound =
                    f_obj.marginalGainWithMemoization(greedySet, candidate_id, false);
                if (verbose)
                    std::cout << "Updated gain as per updated greedy set = "
                              << newCandidateBound << "\n";
                auto nextElem = std::next(it, 1);
				if(verbose) std::cout << "Next element is: " << (*nextElem).second <<"\n";
                if (newCandidateBound >= (*nextElem).first) {
                    if (verbose)
                        std::cout << "..better than next best upper bound, "
                                     "selecting...\n";
                    best_id = candidate_id;
                    best_val = newCandidateBound;
                    break;
                } else {
                    if (verbose) {
                        std::cout << "... NOT better than next best upper "
                                     "bound, updating...\n";
                        std::cout << "...updating its value in sorted set\n";
                    }
                    //if(verbose) std::cout <<"Before erase, it is pointing to: " << (*it).second <<"\n";
                    sortedGains.erase(*it);
                    //if(verbose) std::cout <<"After erase, it is pointing to: " << (*it).second <<"\n";
                    //it = nextElem;
					//if(verbose) std::cout <<"After resetting to next element, it is pointing to: " << (*it).second <<"\n";
                    sortedGains.insert(std::pair<double, ll>(newCandidateBound, candidate_id));
                    //if(verbose) std::cout <<"After insert, it is pointing to: " << (*it).second <<"\n";
                    it = nextElem;
					//if(verbose) std::cout <<"After insert and fixing iterator, it is pointing to: " << (*it).second <<"\n";
                }
            } else {
                it++;
            }
        }
        if (verbose) {
            //std::cout << "...removing from sorted set\n";
            std::cout << "Next best item to add is " << best_id
                      << " and its value addition is " << best_val << "\n";
        }
        sortedGains.erase(std::pair<double, ll>(candidate_val, candidate_id));

        if ((best_val < 0 && stopIfNegativeGain) ||
            (equals(best_val, 0, 1e-5) && stopIfZeroGain)) {
            break;
        } else {
            f_obj.updateMemoization(greedySet, best_id);
            greedySet.insert(
                best_id);  // greedily insert the best datapoint index of
                           // current iteration of while loop
            greedyVector.push_back(std::pair<ll, double>(best_id, best_val));
            rem_budget -= 1;
            remainingSet.erase(best_id);
            if (verbose) {
                std::cout << "Added element " << best_id << " and the gain is "
                          << best_val << "\n";
                std::cout << "Updated greedySet: ";
                for (int i : greedySet) {
                    std::cout << i << " ";
                }
                std::cout << "\n";
            }
            if(showProgress) {
                //TODO: use py::print
                percent = (int)(((iter+1.0)/N)*100);
                if (percent >= displayNext) {
                    //cout << "\r" << "[" << std::string(percent / 5, (char)254u) << std::string(100 / 5 - percent / 5, ' ') << "]";
                    std::cerr << "\r" << "[" << std::string(percent / 5, '|') << std::string(100 / 5 - percent / 5, ' ') << "]";
                    std::cerr << percent << "%" << " [Iteration " << iter + 1 << " of " << N << "]";
                    std::cerr.flush();
                    displayNext += step;
                }
                iter += 1;
            }
        }
        i += 1;
    }
    return greedyVector;
}
