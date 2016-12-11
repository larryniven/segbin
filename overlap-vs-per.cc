#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

std::vector<std::pair<int, int>> align(
    std::vector<int> const& p1,
    std::vector<int> const& p2)
{
    std::vector<std::vector<int>> score;
    std::vector<std::vector<int>> backpointer;

    score.resize(p1.size() + 1);
    for (auto& v: score) {
        v.resize(p2.size() + 1);
    }

    backpointer.resize(p1.size() + 1);
    for (auto& v: backpointer) {
        v.resize(p2.size() + 1);
    }

    score[0][0] = 0;
    backpointer[0][0] = -1;

    for (int i = 1; i < score.size(); ++i) {
        score[i][0] = -i;
        backpointer[i][0] = 1;
    }

    for (int j = 1; j < score[0].size(); ++j) {
        score[0][j] = -j;
        backpointer[0][j] = 2;
    }

    for (int i = 1; i < score.size(); ++i) {
        for (int j = 1; j < score[i].size(); ++j) {
            int max = -std::numeric_limits<int>::max();
            int argmax = -1;

            if (score[i-1][j-1] + (p1[i-1] == p2[j-1] ? 0 : -1) > max) {
                // sub
                max = score[i-1][j-1] + (p1[i-1] == p2[j-1] ? 0 : -1);
                argmax = 0;
            }

            if (score[i-1][j] -1 > max) {
                // ins
                max = score[i-1][j] - 1;
                argmax = 1;
            }

            if (score[i][j-1] - 1 > max) {
                // del
                max = score[i][j-1] - 1;
                argmax = 2;
            }

            score[i][j] = max;
            backpointer[i][j] = argmax;
        }
    }

    int i = score.size() - 1;
    int j = score[i].size() - 1;

    std::vector<std::pair<int, int>> result;

    while (i != 0 && j != 0) {
        if (backpointer[i][j] == 0) {
            result.push_back(std::make_pair(p1[i-1], p2[j-1]));
        } else if (backpointer[i][j] == 1) {
            result.push_back(std::make_pair(-1, p2[j-1]));
        } else if (backpointer[i][j] == 2) {
            result.push_back(std::make_pair(p1[i-1], -1));
        }
    }
}

struct learning_env {

    std::ifstream lat_batch;
    std::ifstream gt_batch;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;
    std::vector<int> sils;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "overlap-vs-per",
        "Learn segmental CRF",
        {
            {"lat-batch", "", true},
            {"gt-batch", "", true},
            {"label", "", true},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    lat_batch.open(args.at("lat-batch"));
    gt_batch.open(args.at("gt-batch"));

    label_id = util::load_label_id(args.at("label"));
    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        id_label[p.second] = p.first;
    }

    if (ebt::in(std::string("sils"), args)) {
        for (auto& s: ebt::split(args.at("sils"))) {
            sils.push_back(label_id.at(s));
        }
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 0;

    segcost::overlap_cost<int> cost_func { sils };

    while (1) {

        ilat::fst lat = ilat::load_lattice(lat_batch, label_id);
        std::vector<segcost::segment<int>> gt_segs = util::load_segments(gt_batch, label_id);

        if (!lat_batch || !gt_batch) {
            break;
        }

        std::cout << lat.data->name << std::endl;

        fst::forward_k_best<ilat::fst> k_best;

        std::vector<int> topo_order = fst::topo_order(lat);

        k_best.first_best(lat, topo_order);

        int f = lat.finals()[0];

        double cost_sum = 0;

        int k = 0;
        while (k < 100) {
            if (k >= k_best.vertex_extra.at(f).deck.size()) {
                break;
            }

            std::vector<int> edges = k_best.best_path(lat, f, k);

            // calculate overlap cost

            double cost = 0;

            for (auto& e: edges) {
                segcost::segment<int> s { lat.tail(e), lat.head(e), lat.output(e) };

                cost += cost_func(gt_segs, s);
            }

            cost_sum += cost;

            // calculate per

            ++k;

            k_best.next_best(lat, f, k);
        }

        std::cout << "paths: " << k << " avg cost: " << cost_sum / k << std::endl;

        ++i;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

}

