#include <stdlib.h>
#include <omp.h>
#include <map>
#include <vector>
#include <string.h>
#include <algorithm>

#pragma once

int parse_uint(const char *str, unsigned int *output) {
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

struct cache_flusher {
    const size_t bigger_than_cachesize;
    float *p;
    float m_init;

    cache_flusher(int megabyte, float init) : 
        bigger_than_cachesize((megabyte*1024*1024)/sizeof(float)), 
        p(new float[bigger_than_cachesize]), m_init(init) { 
        std::cout << "created cache_flusher with " << bigger_than_cachesize << " elements" << std::endl;
        std::cout << "total size: " << megabyte << std::endl;
    }

    ~cache_flusher() {
        delete [] p;
    }

    void flush() {
        #pragma omp parallel for
        for(int i = 0; i < bigger_than_cachesize; i++) {
            p[i] = m_init;
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
        std::sort(st[s].begin(), st[s].end());
        return st[s].front();
    }

    double max(std::string const& s) {
        std::sort(st[s].begin(), st[s].end());
        return st[s].back();
    }

    double mean(std::string const& s) {
        double sum = std::accumulate(st[s].begin(), st[s].end(), 0.0);
        return (sum/st[s].size());
    }

    double sum(std::string const& s) {
        return std::accumulate(st[s].begin(), st[s].end(), 0.0);
    }

    double median(std::string const& s) {
        int size = st[s].size();
        bool odd = size%2;
        if(odd) return st[s][(size+1)/2];
        return (st[s][size/2] + st[s][size/2+1])/2;
    }

    int size(std::string const& s) {
        return st[s].size();
    }

    double rms(std::string const& s) {
        double sum = 0.;
        for(auto e : st[s]) sum += e*e;
        sum /= st[s].size();
        return sqrt(sum);
    }

};
