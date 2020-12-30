/*
constructor() {
    Input: (bool) is master set same as ground set (separateMaster)
    Input: (int) number of elements in the ground set (nv)
    Input: (int) number of elements in the master set (nu)
    Input: (string) mode - normal | sparse | clustered (mode)
    Input: (vector<vector<float>>) dense similarity matrix kernel of size U X V (k_dense)
    Input: sparse similarity matrix kernel (k_sparse)
    Input: number of neighbors (num_neighbors)
    Input: clustering (clusters)
    Input: vector of dense kernels (one for each cluster)
    Input: (bool) partial
    Input: (set) items in the partial ground set (ground_subset)

    asserts {
        if separateMaster, number of elements in the master set should be specified
        if separateMaster, mode can not be sparse or clustered
        if mode is sparse, sparse kernel and num_neighbors must be specified and its dimension should be nv X num_neighbors
        if mode is clustered, clustering and separate dense kernels for each cluster must be specified
        if mode is normal and separateMaster == false, dense kernel of dimension nv X nv must be specified
        if mode is normal and separateMaster == true, dense kernel of nu X nv must be specified
        if partial, separateMaser must be false
    }

    if partial == true {
        effectiveGroundSet = ground_subset
        numEffectiveGroundset = num elements in effectiveGroundSet
    } else {
        effectiveGroundSet = groundSet
        numEffectiveGroundset = num elements in effectiveGroundSet
    }

    if mode == normal and separateMaster == false {
        similarityWithNearestInEffectiveX = vector(numEffectiveGroundset)
        similarityWithNearestInEffectiveX = 0
    } else if mode == normal and separateMaster == true {
        similarityWithNearestInEffectiveX = vector(nu)
        similarityWithNearestInEffectiveX = 0
    } else if mode == sparse {
        similarityWithNearestInEffectiveX = vector(numEffectiveGroundset)
        similarityWithNearestInEffectiveX = 0
    } else if mode == clustered {
        similarityWithNearestInEffectiveX = vector(numEffectiveGroundset)
        similarityWithNearestInEffectiveX = 0
    } else {
        invalid mode
    }
}

evaluate() {
    Input: X

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }

    if mode == normal and separateMaster == false {
        result = 0
        for each element i in effectiveGroundSet {
            find max similarity of i with all items in effectiveX
            result += max
        }
    } else if mode == sparse {

    } else if mode == clustered {
        result = 0
        for each cluster ci {
            relevantSubset = intersect(X, ci)
            if relevantSubset = empty {
                continue
            }
            for each element i in ci {
                find max similarity of i with all items in relevantSubset
                result += max
            }
        }
    } else {
        not possible
    }
}

evaluateSequential() {
    //assumes that pre computed statistics exist for effectiveX
    Input: X

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }
    
    if mode == normal {
        result = 0
        for each element i in effectiveGroundSet {
            result += similarityWithNearestEffectiveX[i]
        }
    } else if mode == sparse {

    } else if mode == clustered {

    } else {
        not possible
    }
    
    
}

marginalGain() {
    Input: X
    Input: item

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }

    if item not in effectiveGroundSet {
        return 0
    }

    if effectiveX contains item {
        return 0
    }

    gain = 0

    if mode == normal {
        for each element i in effectiveGroundSet {
            max = 0
            find max similarity of i with all items in effectiveSubset
            if sim(i, item) > max {
                gain += sim(i, item) - max
            }
        }
        return gain

    } else if mode == sparse {

    } else if mode == clustered {

    } else {
        not possible
    }

}

marginalGainSequential() {
    //assumes that pre computed statistics exist for effectiveX
    Input: X
    Input: item

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }

    if item not in effectiveGroundSet {
        return 0
    }

    if effectiveX contains item {
        return 0
    }

    gain = 0

    if mode == normal {
        for each element i in effectiveGroundSet {
            if sim(i, item) > similarityWithNearestEffectiveX[i] {
                gain += sim(i, item) - similarityWithNearestEffectiveX[i]
            }
        }
        return gain

    } else if mode == sparse {

    } else if mode == clustered {

    } else {
        not possible
    }
}

sequentialUpdate() {
    Input: X
    Input: item

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }

    if mode == normal {
        if item not in effectiveGroundSet {
            return
        }
        for each element i in effectiveGroundSet {
            if similarityWithNearestEffectiveX[i] < sim(i, item) {
                similarityWithNearestEffectiveX[i] = sim(i, item)
            } 
        }
    } else if mode == sparse {

    } else if mode == clustered {

    } else {

    }
}

getEffectiveGroundSet() {
    return effectiveGroundSet
}
*/