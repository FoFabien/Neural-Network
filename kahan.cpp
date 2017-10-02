#include "kahan.hpp"

#include <algorithm>

KahanAccumulation KahanSum(KahanAccumulation accumulation, double value)
{
    KahanAccumulation result;
    double y = value - accumulation.correction;
    double t = accumulation.sum + y;
    result.correction = (t - accumulation.sum) - y;
    result.sum = t;
    return result;
}

KahanAccumulation DoubleSum_K(const std::vector<double> &values)
{
    KahanAccumulation init = {0};
    return std::accumulate(values.begin(), values.end(), init, KahanSum);
}

double DoubleSum(const std::vector<double> &values)
{
    return DoubleSum_K(values).sum;
}

double DoubleSum(const double &a, const double &b)
{
    std::vector<double> values = {a, b};
    std::sort(values.begin(), values.end(), std::greater<double>());
    return DoubleSum_K(values).sum;
}
