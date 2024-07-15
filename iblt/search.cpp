#include <iostream>
#include <map>
#include <random>
#include <set>
#include <array>

#define BUCKETS

// g++ -g -O3 search.cpp -Wno-c++11-extensions -o search 

//largest K to search for
constexpr uint MAX_K = 10;


class hypergraph {
  public:
    using Edge = std::array<uint, MAX_K>;

    std::set<Edge> hg;
    const uint     items;
    const uint     rows;
    const uint     k;

    hypergraph(uint m, uint k, uint n) : items(n), rows(m), k(k) {
        std::random_device         r;
        std::default_random_engine e1(r());

        //assert(MAX_K>=k);
        auto uniform_dist = std::uniform_int_distribution<>(
                0,
#ifdef BUCKETS
                rows / k - 1
#else
                rows - 1
#endif
        );

        for (uint i = 0; i < items; ++i) {
            Edge edge;
            for (uint j = 0, bucket = 0; j < k; ++j, bucket += rows / k) {
                uint result = uniform_dist(e1);
                edge.at(j) = result
#ifdef BUCKETS
                             + bucket
#endif
                ;
            }
            hg.insert(edge);
        }
    };

    // For debugging
    void print_hg() const {
        for (auto edge: hg) {
            print_edge(edge);
        }
    };

    // For debugging
    void print_edge(Edge edge) const {
        std::cout << "[";
        for (auto v: edge) {
            std::cout << v << " ";
        }
        std::cout << "]\n";
    }


    // Peel a hypergraph. This is our main function
    void peel() {
        if (hg.empty()) { return; }

        /* Count the number of times we see each edge... looking for singles */
        std::map<uint, uint> vertex_cnt;
        for (auto edge: hg) {
            for (uint j = 0; j < k; j++) {
                vertex_cnt[edge[j]] += 1;
            }
        }

        std::set<uint> remove_vert;
        for (auto v2c: vertex_cnt) {
            if (v2c.second == 1) {
                remove_vert.insert(v2c.first);
            }
        }

#ifdef BUCKETS
        for (auto vert: remove_vert) {//for each vertex r
            uint r_bucket = floor(vert / (rows / k));
            //assert(r_bucket < k);
            auto i_edge = hg.begin();
            while (i_edge != hg.end()) { //search each edge for r
                if (i_edge->at(r_bucket) == vert) {
                    hg.erase(i_edge);
                    i_edge = hg.end();
                    break; // there is only one, so we can stop looking for it
                }
                if (i_edge != hg.end())
                    i_edge++;
            }
        }
#else
        // if we have buckets, it's a little harder
        for (auto vert: remove_vert) {//for each vertex r
            //assert(vert<rows);
            auto itr = hg.begin();
            while (itr != hg.end()) { //search each edge for r
                for (uint v = 0; v < k; v++) {
                    if (itr->at(v) == vert) {
                        itr = hg.erase(itr);
                        v = k;
                        // break;
                        // itr=hg.end();
                        // break; // there is only one, so we can stop looking for it
                    }
                }
                if (itr != hg.end())
                    itr++;
            }
        }
#endif
    }; //peel

    // Check if a hypergraph decoded
    bool check_decode() {
        uint len = hg.size();

        if (items > len)
            return false;

        uint last_len;
        do {
            last_len = len;
            peel();
            len = hg.size();
        } while (len < last_len);

        return hg.empty();
    };
};



// Determine the decode rate using at least 5000 trials.
// Trials stop when 95% confidence interval is met, or 
// mean is within .99999 to 1.00001  """

int main(int argc, char* argv[]) {
    int         entries  = atoi(argv[1]);
    int         k        = atoi(argv[2]);
    int         rows     = atoi(argv[3]);
    long double goal     = 1.0 * atoi(argv[4]) / atoi(argv[5]);
    long double ci_limit = (1 - goal) / 5;
    long double prob;
    long double ci;
    int         success  = 0;
    int         trials   = 0;
    bool        not_done = true;
    while (not_done == true) {
        for (int sub = 0; sub < 100; sub++, trials++) {
            // std::cout<<"\nNew Trial\n";
            hypergraph h = hypergraph(rows, k, entries);
            if (h.check_decode())
                success += 1;
        }
        prob = 1.0 * success / trials;
        if (success < trials) {
            ci = 1.96 * sqrt(prob * (1.0 - prob) / trials);
        } else {
            ci = -(exp(log(.05) / trials) - 1);
        }
        if (trials >= 5000) {
            if (prob - ci > goal) {
                not_done = false;
                //std::cerr<<"done 1 "<<prob<<" -"<<ci<<">"<<goal <<"\n";
            }
            if (prob + ci <= goal) {
                not_done = false;
                //std::cerr<<"done 2\n";
            }
            if ((prob - ci > goal - ci_limit) && (prob + ci < goal + ci_limit)) {
                not_done = false;
                //std::cerr<<"done 3\n";
            }
        }
    }

    std::cout << success << "," << trials << "\n";
}
