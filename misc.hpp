#ifndef MISC_HPP
#define MISC_HPP

#include <string>

// source : https://stackoverflow.com/questions/4643641/best-way-to-output-a-full-precision-double-into-a-text-file

std::string doubleToText(const double& d);
double textToDouble(const std::string& str);

#endif // MISC_HPP
