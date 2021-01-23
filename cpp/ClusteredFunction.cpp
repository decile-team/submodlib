/*
constructor() {
    Input: (int) number of elements in the ground set (n)
    Input: (vector of sets) c1, c2, ... cm (clustering)
    Input: (string) name of the function, say "FacilityLocation"
    Input: (vector<vector<vector<float>>>) dense similarity matrix kernels for each cluster (k_dense)

    (vector of functions) funcs;

    for each cluster ci {
        Instantiate f with ci (appropriately in partial mode)
        funcs.push_back(fi)
    }

    //possibly construct a vector clusterIDs such that clusterId[i] gives the cluster id where i belongs
}

evaluate() {
    Input: subset to be evaluated (X)
    result = 0;
    for i = 1 to m {
        result += fi.evaluate(X)
    }
    return result;
}

evaluateWithMemoization() {
    //assumes necessary memoization is in place for X 
    Input: subset to be evaluated (X)
    result = 0;
    for i = 1 to m {
        result += fi.evaluateWithMemoization(X)
    }
    return result;
}

marginalGain() {
    Input: X 
    Input: item

    Find which cluster ck item belongs to (ck = clusterIDs[item])
    return fk.marginalGain(X, item)
}

marginalGainWithMemoization() {
    Input: X 
    Input: item

    Find which cluster ck item belongs to (ck = clusterIDs[item])
    return fk.marginalGainWithMemoization(X, item)
}

updateMemoization() {
    Input: X
    Input: item

    Find which cluster ck item belongs to (ck = clusterIDs[item])
    fk.updateMemoization(X, item)
}
*/
