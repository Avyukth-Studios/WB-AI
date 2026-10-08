#include "AIcore.h"
#include <iostream>

void Mini_AI::trainLAgebra(int epohs, int lr)
{
    for(auto& i : inputs) //gets the inputs vector without copying;
    {
        if(i.isEmpty) //checks if input is empyty
        {
            std::string& LastError = "Error: 1910, Input not entered"; //if it is. Log it;
            std::cerr << "404";
        }  

        else
        {  
            for(int e = 1; e >= epohs, ++e)
              {

                  
        


