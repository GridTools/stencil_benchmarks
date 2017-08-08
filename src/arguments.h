#pragma once

#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class arguments_map {
    template <class Type>
    struct overload {};
    using map_t = std::map<std::string, std::string>;

  public:
    using const_iterator = typename map_t::const_iterator;

    std::string get_raw(const std::string &name) const;

    template <class Type = std::string>
    Type get(const std::string &name) const {
        return get_impl(name, overload<Type>());
    }

    bool get_flag(const std::string &name) const;

    const_iterator begin() const { return m_map.begin(); }
    const_iterator end() const { return m_map.end(); }

    arguments_map with(const std::vector<std::pair<std::string, std::string>> &args) const;

  private:
    std::string get_impl(const std::string &name, overload<std::string>) const;
    int get_impl(const std::string &name, overload<int>) const;
    float get_impl(const std::string &name, overload<float>) const;
    double get_impl(const std::string &name, overload<double>) const;

    map_t m_map;
    std::set<std::string> m_flags;

    friend class arguments;
};

std::ostream &operator<<(std::ostream &out, const arguments_map &argsmap);

class arguments {
    struct argument {
        std::string name;
        std::string description;
        std::string default_value;
    };

    struct flag {
        std::string name;
        std::string description;
    };

  public:
    arguments(const std::string &command_name = "command", const std::string &subcommand_name = "subcommand");

    arguments &add(const std::string &name, const std::string &description, const std::string &default_value = "");
    arguments &add_flag(const std::string &name, const std::string &description);

    arguments &command(const std::string &command_name, const std::string &subcommand_name = "subcommand");

    arguments_map parse(int argc, char **argv) const;

  private:
    void print_help() const;

    std::string m_command_name, m_subcommand_name;
    std::vector<argument> m_args;
    std::vector<flag> m_flags;
    std::map<std::string, std::unique_ptr<arguments>> m_command_map;
};
