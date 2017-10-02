#include "misc.hpp"
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <limits>

std::string doubleToText(const double& d)
{
    std::stringstream ss;
    ss << std::setprecision(std::numeric_limits<double>::max_digits10);
    ss << d;
    return ss.str();
}

double textToDouble(const std::string& str)
{
    return strtod(str.c_str(), NULL);
}
