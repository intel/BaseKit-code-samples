//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include<iostream>
#include<vector>
using namespace std;
class Complex2 {
private: 
    int real,imag;
public:
    Complex2(){
        real = 0;
        imag = 0;        
    }
    Complex2(int x, int y){
        real = x;
        imag = y;
    }

    //Overloading the  == operator
    friend bool operator==(const Complex2 &a, const Complex2 &b)
    {
        return (a.real == b.real) && (a.imag == b.imag);
    }

    //The function performs Complex number multiplication and returns a Complex2 object. 
    Complex2 complex_mul(const Complex2& obj){
        return Complex2(((real*obj.real) - (imag*obj.imag)),((real*obj.imag)+(imag*obj.real)));
    }

	//Overloading the ostream operator to print the objects of the Complex2 object
    friend ostream& operator << (ostream& out, const Complex2& obj){
        out<<"("<<obj.real<<" : "<<obj.imag<<"i)";
        return out;
    }
};