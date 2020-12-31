/*
base class: Optimizer

this class: NaiveGreedyOptimizer

constructor: nothing

maximize() {
    Input: (SetFunction) The function to maximize (f)
    Input: (double) budget
    Input: (bool) stopIfZeroGain (do not add items if gain is 0)
    Input: (bool) stopIfNegativeGain (do not add items if gain is negative)
    Input: (bool) verbosity
    Output: (std::vector<std::pair<int, double>>) greedyVector - ordered list of element and the gain associated with that element
    
    //LATER: take care of equal guys later
    //LATER: take care of different sizes of each items - becomes a candidate only if best and within budget, cost sensitive selection
    
    greedyVector = empty
    greedySet = empty
    remainingBudget = budget
    iter = 0
    while remainingBudget > 0 {
        iter++
        currValBest = -1 * std::numeric_limits<double>::max();
        currIdBest = -1
        for each element i in f.getEffectiveGroundSet() {
            if i in greedySet {
                continue
            }
            gain = f.marginalGainSequential(greedySet, i)
            if gain > currValBest { 
                currValBest = gain;
                currIdBest = i
            } 
        } 
        //found the next best guy
        if (currValBest < 0 && stopIfNegativeGain) || (currValBest == 0 && stopIfZeroGain) {
            break
        } else {
            //add this guy and proceed to next iteration
            f.sequentialUpdate(greedySet, currIdBest) 
            greedySet.insert(currIdBest)
            greedyVector.push_back(currIdBest, currValBest)
            remainingBudget -= 1
        }
    }
}
*/
