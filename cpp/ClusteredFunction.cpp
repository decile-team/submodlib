/*
constructor() {
    Input: (int) number of elements in the ground set (n)
    Input: (vector of sets) c1, c2, ... cm (clustering)
    Input: (string) name of the function, say "FacilityLocation"
    Input: (vector<vector<float>>) dense similarity matrix kernel (k_dense)

    (vector of functions) funcs;

    for each cluster ci {
        Instantiate f with ci
        funcs.push_back(fi)
    }
}

evaluate() {
    Input: subset to be evaluated (X)
    result = 0;
    for i = 1 to m {
        result += fi.evaluate(X)
    }
    return result;
}

evaluateSequential() {
    //assumes necessary memoization is in place for X 
    Input: subset to be evaluated (X)
    result = 0;
    for i = 1 to m {
        result += fi.evaluateSequential(X)
    }
    return result;
}

marginalGain() {
    Input: X 
    Input: item

    Find which cluster ck item belongs to
    return fk.marginalGain(X, item)
}

marginalGainSequential() {
    Input: X 
    Input: item

    Find which cluster ck item belongs to
    return fk.marginalGainSequential(X, item)
}

sequentialUpdate() {
    Input: X
    Input: item

    for i = 1 to m {
        fi.sequentialUpdate(X, item)
    }
}
*/
