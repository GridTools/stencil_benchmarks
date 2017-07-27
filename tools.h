#ifdef FLAT_MODE
#include <hbwmalloc.h>
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

#include <omp.h>

#pragma once

int parse_uint(const char *str, unsigned int *output) {
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

class cache_flusher {
  public:
    void flush() {
        // just wait, this should to a context switch
        // and the OS will flush all the cache
        #pragma omp parallel
        {
            std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
        }
    }
};


struct timing {
    std::map<std::string, std::vector<double> > st;

    void insert(std::string const& s, double t) {
        st[s].push_back(t);
    }

    void clear() {
        st.clear();
    }

    double min(std::string const& s) {
        if (!st.count(s)) return 0.0;
        std::sort(st[s].begin(), st[s].end());
        return st[s].front();
    }

    double max(std::string const& s) {
        if (!st.count(s)) return 0.0;
        std::sort(st[s].begin(), st[s].end());
        return st[s].back();
    }

    double mean(std::string const& s) {
        if (!st.count(s)) return 0.0;
        double sum = std::accumulate(st[s].begin(), st[s].end(), 0.0);
        return (sum/st[s].size());
    }

    double sum(std::string const& s) {
        if (!st.count(s)) return 0.0;
        return std::accumulate(st[s].begin(), st[s].end(), 0.0);
    }

    double median(std::string const& s) {
        if (!st.count(s)) return 0.0;
        int size = st[s].size();
        bool odd = size%2;
        if(odd) return st[s][(size+1)/2];
        return (st[s][size/2] + st[s][size/2+1])/2;
    }

    int size(std::string const& s) {
        if (!st.count(s)) return 0.0;
        return st[s].size();
    }

    double rms(std::string const& s) {
        if (!st.count(s)) return 0.0;
        double sum = 0.;
        for(auto e : st[s]) sum += e*e;
        sum /= st[s].size();
        return sqrt(sum);
    }

};
