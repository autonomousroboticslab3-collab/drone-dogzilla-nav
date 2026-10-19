/* Hello, C++! */
// This is my first C++ program
 
#include <iostream>
using namespace std;




int abc3(int ans){

ans=ans+2;
return ans;

}

void abc1(int *p){
*p=*p+2;
}


void abc2(int &p){
    int &a3=p;

    a3=a3+5;
    cout << "abc2 " << p << endl;
}

int main()
{
    int a=10;
    int *point;

    cout << "Hello, C++!" << endl;
    point=&a;
    //
    cout << *point <<" "<< point << endl;
    cout << &a<<" "<< a << endl;
    //
    abc1(&a);
    cout << "abc1 " <<a << endl;
    //
    int a2=2;
    abc2(a2);
    cout << a2 << endl;




    return 0;
}