#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "seg/fscrf.h"
#include <fstream>

struct oracle_env {

    std::ifstream gold_batch;

    std::ifstream lattice_batch;

    fscrf::learning_args l_args;

    std::unordered_map<std::string, std::string> args;

    oracle_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "oracle-cost",
        "Learn segmental CRF",
        {
            {"gold-batch", "", true},
            {"lattice-batch", "", true},
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

    oracle_env env { args };

    env.run();

    return 0;
}

oracle_env::oracle_env(std::unordered_map<std::string, std::string> args)
{
    gold_batch.open(args.at("gold-batch"));

    lattice_batch.open(args.at("lattice-batch"));

    l_args.label_id = util::load_label_id(args.at("label"));

    l_args.id_label.resize(l_args.label_id.size());
    for (auto& p: l_args.label_id) {
        l_args.labels.push_back(p.second);
        l_args.id_label[p.second] = p.first;
    }
}

void oracle_env::run()
{
    ebt::Timer timer;

    int i = 1;

    double min_cost_sum = 0;
    double max_cost_sum = 0;
    double frames_sum = 0;

    while (1) {

        ilat::fst lat = ilat::load_lattice(lattice_batch, l_args.label_id);
        std::vector<segcost::segment<int>> gold_segs = util::load_segments(gold_batch, l_args.label_id);

        if (!lattice_batch || !gold_batch) {
            break;
        }

        fscrf::fscrf_data graph_data;
        graph_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
            std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gold_segs, l_args.sils)),
            -1));

        graph_data.weight_func = graph_data.cost_func;

        fscrf::fscrf_fst scrf { graph_data };

        fst::forward_one_best<fscrf::fscrf_fst> min_one_best;

        for (auto& i: scrf.initials()) {
            min_one_best.extra[i] = fst::forward_one_best<fscrf::fscrf_fst>::extra_data {-1, 0};
        }

        min_one_best.merge(scrf, *graph_data.topo_order);

        std::vector<int> min_path = min_one_best.best_path(scrf);

        double min_cost = 0;
        for (auto& e: min_path) {
            min_cost -= scrf.weight(e);
        }

        graph_data.cost_func = std::make_shared<scrf::seg_cost<ilat::fst>>(
            scrf::make_overlap_cost<ilat::fst>(gold_segs, l_args.sils));

        graph_data.weight_func = graph_data.cost_func;

        fst::forward_one_best<fscrf::fscrf_fst> max_one_best;

        for (auto& i: scrf.initials()) {
            max_one_best.extra[i] = fst::forward_one_best<fscrf::fscrf_fst>::extra_data {-1, 0};
        }

        max_one_best.merge(scrf, *graph_data.topo_order);

        std::vector<int> max_path = max_one_best.best_path(scrf);

        double max_cost = 0;
        for (auto& e: max_path) {
            max_cost += scrf.weight(e);
        }

        std::cout << i << ": min cost: " << min_cost << " max cost: " << max_cost << " frames: " << lat.time(lat.finals().front())
            << " min er: " << min_cost / lat.time(lat.finals().front())
            << " max er: " << max_cost / lat.time(lat.finals().front()) << std::endl;

        min_cost_sum += min_cost;
        max_cost_sum += max_cost;
        frames_sum += lat.time(lat.finals().front());

        ++i;
    }

    std::cout << "total min cost: " << min_cost_sum << " total max cost: " << max_cost_sum << " total frames: " << frames_sum
        << " min er: " << min_cost_sum / frames_sum
        << " max er: " << max_cost_sum / frames_sum << std::endl;

}

