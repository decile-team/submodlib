/*
constructor() {
    Input: (bool) is master set same as ground set (separateMaster)
    Input: (int) number of elements in the ground set (nv)
    Input: (int) number of elements in the master set (nu)
    Input: (string) mode - dense | sparse | clustered (mode)
    Input: (vector<vector<float>>) dense similarity matrix kernel of size U X V (k_dense)
    Input: sparse similarity matrix kernel (k_sparse)
    Input: number of neighbors (num_neighbors)
    Input: (vector<set>) clustering (clusters)
    Input: vector of dense kernels (one for each cluster)
    Input: (bool) partial  #only to be used with ClusteredFunction
    Input: (set) items in the partial ground set (ground_subset)

    asserts {
        if separateMaster, number of elements in the master set should be specified
        if separateMaster, mode can not be sparse or clustered
        if mode is sparse, sparse kernel and num_neighbors must be specified and its dimension should be nv X num_neighbors
        if mode is clustered, clustering and separate dense kernels for each cluster must be specified
        if mode is dense and separateMaster == false, dense kernel of dimension nv X nv must be specified
        if mode is dense and separateMaster == true, dense kernel of nu X nv must be specified
        if partial, separateMaster must be false
        if partial, mode must be dense or sparse
        ground_subset cannot have elements outside 0 to nv-1
    }

    create groundSet with items 0 to nv-1

    if partial == true {
        effectiveGroundSet = ground_subset
    } else {
        effectiveGroundSet = groundSet
    }
    numEffectiveGroundset = num elements in effectiveGroundSet

    if mode == dense and separateMaster == true {
        numMaster = nu
        create master set with items 0 to nu-1
    } else {
        numMaster = numEffectiveGroundset
        masterSet = clone of effectiveGroundSet
    }
    if mode == clustered {
        clusterIDs = vector of size groundSet with clusterId[i] = cluster id where i belongs
        relevantX = vector of sets (size = number of clusters)
        clusteredSimilarityWithNearestInRelevantX = vector(numGroundset)
        for each cluster ci {
            create an empty set
            pushback that set in relevantX
            // create a vector of size = num elements in ci
            // initialize it to 0s
            // pushack this vector to clusteredSimilarityWithNearestInRelevantX
        }
        clusteredSimilarityWithNearestInRelevantX = 0
    } else {
        similarityWithNearestInEffectiveX = vector(numMaster)
        similarityWithNearestInEffectiveX = 0
    }
}
    
evaluate() {
    Input: X

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }
    result = 0
    if mode == clustered {
       for each cluster ci {
            relevantSubset = intersect(X, ci)
            if relevantSubset = empty {
                continue
            }
            for each element i in ci {
                max = 0
                find max similarity of i with all items in relevantSubset
                result += max
            }
        }
    } else {
        for each element i in masterSet {
            max = 0
            find max similarity of i with all items in effectiveX
            result += max
        }
    }
    return result
}

evaluateSequential() {
    //assumes that appropriate pre computed statistics exist for effectiveX
    Input: X

    if partial == true {
        effectiveX = intersect(X, effectiveGroundSet)
    } else {
        effectiveX = X
    }

    result = 0
    if mode == clustered {
       for each cluster ci {
            if relevantX[ci] == empty {
                continue
            }
            for each element i in ci {
                result += clusteredSimilarityWithNearestInRelevantX[i]
            }
        }
    } else {
        for each element i in masterSet {
            result += similarityWithNearestInEffectiveX[i]
        }
    }
    return result
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
    
    if mode == clustered {
        ci = clusterIDs[item]
        relevantSubset = intersect(ci, X)
        if relevantSubset == empty {
            for each element i in ci {
                gain += sim(i, item)
            }
        } else {
            for each element i in ci {
                max = 0
                find max similarity of i with all items in relevantSubset
                if sim(i, item) > max {
                    gain += sim(i, item) - max
                } 
            }
        }
    } else {
        for each element i in masterSet {
            max = 0
            find max similarity of i with all items in effectiveSubset
            if sim(i, item) > max {
                gain += sim(i, item) - max
            }
        }
    }
    return gain
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
    
    if mode == clustered {
        ci = clusterIDs[item]
        relevantSubset = relevantX[ci]
        if relevantSubset == empty {
            for each element i in ci {
                gain += sim(i, item)
            }
        } else {
            for each element i in ci {
                if sim(i, item) > clusteredSimilarityWithNearestInRelevantX[i] {
                    gain += sim(i, item) - clusteredSimilarityWithNearestInRelevantX[i]
                } 
            }
        }
    } else {
        for each element i in masterSet {
            if sim(i, item) > similarityWithNearestEffectiveX[i] {
                gain += sim(i, item) - similarityWithNearestEffectiveX[i]
            }
        }
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
    
    if mode == clustered {
        ci = clusterIds[item]
        //update clusteredSimilarityWithNearestInRelevantX
        for each element i in ci {
            if sim(i, item) > clusteredSimilarityWithNearestInRelevantX[i] {
                clusteredSimilarityWithNearestInRelevantX[i] = sim(i, item)
            }
        }
        //update relevantX
        relevantX[ci].insert(item)
    } else {
        if item not in effectiveGroundSet or item in X {
            return
        }
        for each element i in masterSet {
            if similarityWithNearestEffectiveX[i] < sim(i, item) {
                similarityWithNearestEffectiveX[i] = sim(i, item)
            } 
        }
    }
}

getEffectiveGroundSet() {
    return effectiveGroundSet
}
*/