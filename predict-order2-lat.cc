#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct inference_env {

    std::ifstream frame_batch;
    std::ifstream lat_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> features;

    std::shared_ptr<ilat::fst> lm;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;
    std::vector<int> sils;

    std::unordered_map<std::string, std::string> args;

    inference_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-order2-lat",
        "Decode with segmental CRF",
        {
            {"frame-batch", "", false},
            {"lat-batch", "", true},
            {"lm", "", true},
            {"param", "", true},
            {"features", "", true},
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

    inference_env env { args };

    env.run();

    return 0;
}

inference_env::inference_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    lat_batch.open(args.at("lat-batch"));

    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    features = ebt::split(args.at("features"), ",");

    param = fscrf::make_tensor_tree(features);
    tensor_tree::load_tensor(param, args.at("param"));

    label_id = util::load_label_id(args.at("label"));
    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        id_label[p.second] = p.first;
    }

    std::ifstream lm_stream { args.at("lm") };
    lm = std::make_shared<ilat::fst>(ilat::load_arpa_lm(lm_stream, label_id));
    lm_stream.close();

    if (ebt::in(std::string("sils"), args)) {
        for (auto& s: ebt::split(args.at("sils"))) {
            sils.push_back(label_id.at(s));
        }
    }
}

void inference_env::run()
{
    ebt::Timer timer;

    int i = 0;

    while (1) {

        ilat::fst lat = ilat::load_lattice(lat_batch, label_id);

        if (!lat_batch) {
            break;
        }

        std::vector<std::vector<double>> frames;
        if (ebt::in(std::string("frame-batch"), args)) {
            frames = speech::load_frame_batch(frame_batch);
        }

        ilat::add_eps_loops(lat);

        ilat::lazy_pair_mode1 composed_fst { lat, *lm };

        fscrf::fscrf_pair_data graph_data;
        graph_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(fst::topo_order(composed_fst));
        graph_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        graph_data.weight_func = fscrf::make_pair_weights(features, var_tree, frames);

        std::shared_ptr<ilat::pair_fst> path = scrf::shortest_path(graph_data);

        for (auto& e: path->edges()) {
            if (path->output(e) == 0) {
                continue;
            }

            std::cout << id_label.at(path->output(e)) << " ";
        }
        std::cout << "(" << lat.data->name << ".dot)" << std::endl;

        ++i;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

}

