/*
naiveGreedyMax() {
    Input: SetFunction (f)
    Input: budget
    Output: greedyVector
    
    greedyVector = empty
    greedySet = empty
    remainingBudget = budget
    while remainingBudget > 0 {
        best
        for each element i in f.getEffectiveGroundSet() {
            if i in greedySet {
                continue
            }
            if f.marginalGainSequential(greedySet, i) > best {
                best = i;
            } 
        } 
        f.sequentialUpdate(greedySet, best) 
        greedySet.insert(best)
        greedyVector.push_back(best)
        remainingBudget -= 1
    }
}
*/
