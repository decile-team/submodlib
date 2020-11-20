#include<iostream>
#include<string>
#include"HelloWorld.h"

mess::mess(std::string s) : m(s) {}

void mess::out()
{
	std::cout << m;
}


