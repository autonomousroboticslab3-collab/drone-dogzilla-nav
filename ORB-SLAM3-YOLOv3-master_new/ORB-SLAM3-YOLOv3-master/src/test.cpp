#include "iostream"
#include<vector>
using namespace std;

int main()
{
    vector<int> a123 = (10, 20, 30, 40);
    vector<int> b123 = (50, 60, 70, 80);

    return 0;
}

// int distance(vector<vector<int>> va, vector<vector<float>> vb)
// {
//     int totalDistance = 0;

//     for(Point imagePoint: image)
//     {
//         int minDistance = numeric_limits<int>::max();

//         for(Point tempPoint: tempImage)
//         {
//             Point diff = imagePoint - tempPoint;
//             int length = (diff.x * diff.x) + (diff.y * diff.y);

//             if(length < minDistance) minDistance = length;
//             if(length == 0) break;
//         }

//         if(minDistance > totalDistance) totalDistance = minDistance;
//     }
//     return totalDistance;
// }