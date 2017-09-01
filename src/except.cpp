#include "except.h"

#include <sstream>

error::error(const std::string &file, int line, const std::string &what) : std::runtime_error(msg(file, line, what)) {}

std::string error::msg(const std::string &file, int line, const std::string &what) {
    std::stringstream s;
    s << line;
    return "ERROR in file '" + file + "' at line " + s.str() + ": " + what;
}
