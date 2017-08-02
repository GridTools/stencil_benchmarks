#pragma once

#include <map>

template <class Product, class Identifier, class Creator>
class factory {
 public:
  void add(const Identifier& id, Creator creator) {
    m_map.emplace(id, creator);
  }

  template <class... Args>
  Product* create(const Identifier& id, Args&&... args) const {
    auto i = m_map.find(id);
    if (i == m_map.end()) return nullptr;
    return (i->second)(std::forward<Args>(args)...);
  }

  bool registered(const Identifier& id) const {
    return m_map.find(id) != m_map.end();
  }

 private:
  std::map<Identifier, Creator> m_map;
};
