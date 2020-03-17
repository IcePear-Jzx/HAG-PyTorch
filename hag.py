def get_V():
    V_set = {1, 2, 3, 4}
    V = {
        1: {2, 3, 4},
        2: {1, 3, 4},
        3: {1, 2},
        4: {1, 2}
    }
    v_num = 4
    return V, V_set, v_num


if __name__ == "__main__":
    capacity = 2
    V, V_set, v_num = get_V()
    V_agg = dict()
    agg_num = 0
    while agg_num < capacity:
        v1 = v2 = r_max = 0
        for i in range(1, v_num+1):
            for j in range(i+1, v_num+1):
                r_ij = len(V[i] & V[j]) # redundancy
                if r_ij > r_max:
                    r_max = r_ij
                    v1 = i
                    v2 = j
        if r_max > 1:
            agg_num += 1
            w = v_num + agg_num
            V[v1].add(w)
            V[v2].add(w)
            U = V[v1] & V[v2] & V_set
            V[v1] -= U
            V[v2] -= U
            V_agg[w] = U
    
    for k, v in V.items():
        print(k, ':', v)

    for k, v in V_agg.items():
        print(k, ':', v)

            


