# given a defined length of a route (how long a vehicle can go), decide how many cities it can pass by.
# -> cities are points represented by (x, y) coordinate.
# -> route is a single value indicating its length
# -> hints: permutate all points and calculate the number (count(cities[])), the computation cost should be n!
import itertools

def treverseMostCities(routeLen, cityList):
    points = {idx: x for idx, x in enumerate(cityList)}
    permutationResultsIter = itertools.permutations(points.keys())
    maxCountCities = 0
    for x in permutationResultsIter:
        dist = 0
        for idx, _ in enumerate(x[:-1]):
            tmpDist = ((points[x[idx]][0] + points[x[idx+1]][0])**2 + (points[x[idx]][1] + points[x[idx+1]][1])**2)**0.5
            if (dist + tmpDist > routeLen):
                break
            else:
                dist += tmpDist
        if (idx+1 > maxCountCities):
            maxCountCities = idx + 1
    return maxCountCities

print(treverseMostCities(10, [[1,2], [3,4], [2,1], [1,1], [0, 0]]))
print(treverseMostCities(2, [[0,0], [1,1], [2,2]]))