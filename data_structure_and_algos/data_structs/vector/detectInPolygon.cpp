// detect if a point is inside a polygon
// Solution: 
// Compute the sum of the angles made between the test point and each pair of orderly sequenced points making up the polygon. 
// If this sum is 2pi then the point is an interior point, if 0 then the point is an exterior point,
// for that,
// an exterior point would see symmetrical angles' sum to zero, in the extreme case where the polygon is a very long rectangle,
//      an exterior point should see half polygon points from -90 degree degrees to 90 degrees, then back from 90 degrees to -90 degrees
// an interior point, such as in a circle, should see angles' sum to 360 degrees going through a full circle.


#include <vector>
#include <math.h>
#include <iostream>

#define PI 3.14159

/*
   Return the angle between two vectors on a plane
   The angle is from vector 1 to vector 2, positive anticlockwise
   The result is between -pi -> pi
*/
double findAngle2D(double x1, double y1, double x2, double y2)
{
   double dtheta,theta1,theta2;

   theta1 = atan2(y1,x1);
   theta2 = atan2(y2,x2);
   dtheta = theta2 - theta1;
   while (dtheta > PI)
      dtheta -= 2 * PI;
   while (dtheta < -PI)
      dtheta += 2 * PI;

   return(dtheta);
}


class Solution {
public:
    static bool detectInPolygon(std::vector<std::pair<int, int>>& polygon, std::pair<int, int>& point)
    {
        int i;
        double angle=0;
        std::pair<double, double> p = point;
        std::pair<double, double> p1,p2;
        int n = polygon.size();

        for (i=0; i < n; i++) {
            p1.first = polygon[i].first - p.first;
            p1.second = polygon[i].second - p.second;
            p2.first = polygon[(i+1) % n].first - p.first;
            p2.second = polygon[(i+1)%n].second - p.second;
            angle += findAngle2D(p1.first,p1.second,p2.first,p2.second);
        }

        if (std::abs(angle) < PI)
            return false;
        else
            return true;
    }
};

int main(){

    std::vector<std::pair<int, int>> triangle{{1, 0}, {5,0}, {5, 50}}; 
    std::pair<int, int> exPoint{0,0}, inPoint1{4,0}, inPoint2{4,1};
    std::vector<std::pair<int, int>> rectangle{{1, 0}, {5,0}, {5, 50}, {1, 50}}; 
    std::vector<std::pair<int, int>> polygon{{1, 0}, {3,0}, {5, 5}, {10, 10},
                                            {15, 10}, {10,15}, {5, 20}, {1, 30} }; 
    std::pair<int, int> inPoint3{7,10};
    
    std::cout << Solution::detectInPolygon(triangle, exPoint) << std::endl;
    std::cout << Solution::detectInPolygon(triangle, inPoint1) << std::endl;
    std::cout << Solution::detectInPolygon(triangle, inPoint2) << std::endl;

    std::cout << Solution::detectInPolygon(polygon, exPoint) << std::endl;
    std::cout << Solution::detectInPolygon(polygon, inPoint3) << std::endl;


    return 0;
}