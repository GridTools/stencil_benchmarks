#include <cstdlib>

#include <getopt.h>

#include "arguments.h"
#include "except.h"

std::string arguments_map::get_raw(const std::string& name) const {
  auto i = m_map.find(name);
  if (i == std::end(m_map))
    throw ERROR("given argument '" + name + "' is not in argument map");
  return i->second;
}

bool arguments_map::get_flag(const std::string& name) const {
  return m_flags.count(name);
}

std::string arguments_map::get_impl(const std::string& name,
                                    overload<std::string>) const {
  return get_raw(name);
}

int arguments_map::get_impl(const std::string& name, overload<int>) const {
  return std::stoi(get_raw(name));
}

float arguments_map::get_impl(const std::string& name, overload<float>) const {
  return std::stof(get_raw(name));
}

double arguments_map::get_impl(const std::string& name,
                               overload<double>) const {
  return std::stod(get_raw(name));
}

arguments_map arguments_map::with(
    const std::vector<std::pair<std::string, std::string>>& args) const {
  arguments_map copy = *this;
  for (auto& a : args) copy.m_map[a.first] = a.second;
  return copy;
}

std::ostream& operator<<(std::ostream& out, const arguments_map& argsmap) {
  out << "Arguments:" << std::endl;
  std::size_t name_maxl = 0;
  std::size_t value_maxl = 0;
  for (const auto& arg : argsmap) {
    name_maxl = std::max(name_maxl, arg.first.size());
    value_maxl = std::max(value_maxl, arg.second.size());
  }
  for (const auto& arg : argsmap)
    out << "    " << std::setw(name_maxl + 1) << std::left << (arg.first + ":")
        << "    " << arg.second << std::endl;
  return out;
}

arguments::arguments(const std::string& command_name,
                     const std::string& subcommand_name)
    : m_command_name(command_name), m_subcommand_name(subcommand_name) {}

arguments& arguments::add(const std::string& name,
                          const std::string& description,
                          const std::string& default_value) {
  m_args.push_back({name, description, default_value});
  return *this;
}

arguments& arguments::add_flag(const std::string& name,
                               const std::string& description) {
  m_flags.push_back({name, description});
}

arguments& arguments::command(const std::string& command_name,
                              const std::string& subcommand_name) {
  return *(m_command_map
               .emplace(command_name, std::unique_ptr<arguments>(new arguments(
                                          command_name, subcommand_name)))
               .first->second);
}

void arguments::print_help() const {
  std::cout << "Command usage: " << m_command_name << " [OPTION]...";
  if (!m_command_map.empty()) {
    std::string subcommand_name = m_subcommand_name;
    for (auto& c : subcommand_name)
      c = std::toupper(static_cast<unsigned char>(c));
  }
  std::cout << std::endl << std::endl;

  std::size_t name_maxl = 0;
  std::size_t default_value_maxl = 0;
  for (const auto& arg : m_args) {
    name_maxl = std::max(name_maxl, arg.name.size());
    default_value_maxl = std::max(default_value_maxl, arg.default_value.size());
  }
  for (const auto& flag : m_flags)
    name_maxl = std::max(name_maxl, flag.name.size());

  std::cout << "Supported options:" << std::endl;
  for (const auto& arg : m_args) {
    std::cout << "    " << std::setw(name_maxl + 2) << std::left
              << ("--" + arg.name) << "    ";
    if (!arg.default_value.empty())
      std::cout << std::setw(default_value_maxl + 11) << std::left
                << ("(default: " + arg.default_value + ")");
    else
      std::cout << std::setw(default_value_maxl + 11) << " ";
    std::cout << "    " << arg.description << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Supported flags:" << std::endl;
  for (const auto& flag : m_flags) {
    std::cout << "    " << std::setw(name_maxl + 2) << std::left
              << ("--" + flag.name) << "    ";
    std::cout << std::setw(default_value_maxl + 11) << " ";
    std::cout << "    " << flag.description << std::endl;
  }
  std::cout << std::endl;

  if (!m_command_map.empty()) {
    std::cout << "Supported commands:" << std::endl;
    for (const auto& command : m_command_map) {
      std::cout << "    " << command.first << std::endl;
    }
  }
}

arguments_map arguments::parse(int argc, char** argv) const {
  std::string command;
  int subcommand_argc = argc;
  for (int i = 0; i < argc; ++i) {
    for (const auto& c : m_command_map) {
      if (c.first == argv[i]) {
        subcommand_argc = i;
        command = c.first;
        break;
      }
    }
  }

  arguments_map argsmap;
  for (const auto& arg : m_args) argsmap.m_map[arg.name] = arg.default_value;

  std::vector<option> options;
  for (const auto& arg : m_args)
    options.push_back({arg.name.c_str(), required_argument, nullptr, 0});
  for (const auto& flag : m_flags)
    options.push_back({flag.name.c_str(), no_argument, nullptr, 0});
  options.push_back({0, 0, 0, 0});

  opterr = 0;
  optind = 1;
  int index;
  while (true) {
    index = -1;
    int c = getopt_long(subcommand_argc, argv, "h", options.data(), &index);

    if (c == -1) break;

    switch (c) {
      case 0:
        if (index < m_args.size())
          argsmap.m_map[m_args[index].name] = optarg;
        else
          argsmap.m_flags.insert(m_flags[index - m_args.size()].name);
        break;
      case 'h':
        print_help();
        std::exit(0);
      case '?':
        std::cerr << "Error: invalid argument '" << optopt << "' for command '"
                  << m_command_name << "'" << std::endl;
      default:
        std::abort();
    }
  }

  if (!m_command_map.empty() && subcommand_argc == argc) {
    std::cerr << "Error: no valid sub-command for command '" << m_command_name
              << "'" << std::endl
              << std::endl;
    print_help();
    std::abort();
  }

  if (optind < subcommand_argc) {
    for (int i = optind; i < subcommand_argc; ++i)
      std::cerr << "Error: invalid " << m_subcommand_name << " '" << argv[i]
                << "'" << std::endl;
    std::abort();
  }

  if (!m_command_map.empty()) {
    argsmap.m_map[m_subcommand_name] = command;
    auto command_argsmap = m_command_map.find(command)->second->parse(
        argc - subcommand_argc, argv + subcommand_argc);
    for (const auto& a : command_argsmap.m_map)
      argsmap.m_map[a.first] = a.second;
    for (const auto& f : command_argsmap.m_flags) argsmap.m_flags.insert(f);
  }

  return argsmap;
}
