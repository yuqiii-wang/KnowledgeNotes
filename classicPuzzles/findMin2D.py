# Find minima in a 2d array, neibghbors refer to up, down left and right cells. 
def find_minima(n, m):
    # The new_m init is deprecated as __mul__ for list is shallow copy of reference,
    # instead [].copy() or [][:] should be used.
        # new_m = [[9999999.0] * (n + 2)] * (n + 2) 
    #
    new_m = [([9999999.0] * (n + 2)).copy() for i in range(n + 2)] # for large val padding
    for row_idx in range(n):
        for col_idx in range(n):
            new_m[row_idx+1][col_idx+1] = m[row_idx][col_idx]
    
    result = []
    for row_idx, _ in enumerate(new_m[1:-1]):
        for col_idx, _ in enumerate(new_m[row_idx+1][1:-1]):
            if new_m[row_idx+1][col_idx+1] < new_m[row_idx+1][col_idx] and \
                new_m[row_idx+1][col_idx+1] < new_m[row_idx+1][col_idx+2] and \
                new_m[row_idx+1][col_idx+1] < new_m[row_idx][col_idx+1] and \
                new_m[row_idx+1][col_idx+1] < new_m[row_idx+2][col_idx+1]:

                result.append(new_m[row_idx+1][col_idx+1])
    if result:
        return sorted(result)[:3]
    else:
        return result

print(find_minima(5, [[5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 1.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 4.0, 5.0], [5.0, 5.0, 4.0, 2.0, 3.0], [0.0, 5.0, 5.0, 3.0, 4.0]]))