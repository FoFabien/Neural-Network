#ifndef KAHAN_HPP
#define KAHAN_HPP

#include <numeric>
#include <iostream>
#include <vector>

// source : https://stackoverflow.com/questions/10330002/sum-of-small-double-numbers-c

struct KahanAccumulation
{
    double sum;
    double correction;
};

KahanAccumulation KahanSum(KahanAccumulation accumulation, double value);

KahanAccumulation DoubleSum_K(const std::vector<double> &values);
double DoubleSum(const std::vector<double> &values);
double DoubleSum(const double &a, const double &b);

#endif // KAHAN_HPP
