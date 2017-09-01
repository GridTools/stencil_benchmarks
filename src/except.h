#pragma once

#include <stdexcept>

class error : public std::runtime_error {
  public:
    error(const std::string &file, int line, const std::string &what);

  private:
    static std::string msg(const std::string &file, int line, const std::string &what);
};

#define ERROR(msg) error(__FILE__, __LINE__, (msg))
