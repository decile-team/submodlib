/*
constructor() {
    Input: (int) number of elements in the ground set (n)
    Input: (string) mode - dense | sparse | clustered (mode)
    Input: (vector<vector<float>>) dense similarity matrix kernel (k_dense)
    Input: sparse similarity matrix kernel (k_sparse)
    Input: number of neighbors (num_neighbors)
    Input: clustering (clusters)
    Input: (bool) partial
    Input: (set) items in the partial ground set (ground_subset)

    if partial == true {
        effectiveGroundSet = ground_subset
        numEffectiveGroundset = num elements in effectiveGroundSet
    } else {
        effectiveGroundSet = groundSet
        numEffectiveGroundset = num elements in effectiveGroundSet
    }
    if mode == dense {
        if k_dense == empty {
            error
            return
        } 
        similarityWithNearestInEffectiveX = vector(numEffectiveGroundset)
        similarityWithNearestInEffectiveX = 0
        
    } else if mode == sparse {
        if k_sparse == empty or num_neighbors == 0 {
            error
            return
        }
    } else if mode == clustered {
        if clusters == empty {
            error
            return
        }
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

    if mode == dense {
        result = 0
        for each element i in effectiveGroundSet {
            find max similarity of i with all items in effectiveSubset
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
    
    if mode == dense {
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

    if mode == dense {
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

    if mode == dense {
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

    if mode == dense {
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