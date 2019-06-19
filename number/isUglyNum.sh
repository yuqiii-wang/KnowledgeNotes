#!/bin/bash

# Write a program to check whether a given number is an ugly number.

# Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.

isUglyNum(){
    num=$1;
    for i in {2..6}
    do
        if [ $num -eq 0 ]; then 
            break;
        fi
        while [ $(($num % $i)) == 0 ];
        do
            num=$(($num / $i));
        done
    done
    
    if [ $num -eq 1 ]; then
        return 1;
    else
        return 0;
    fi
}

isUglyNum 14
echo $?