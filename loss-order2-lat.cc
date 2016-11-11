#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream lat_batch;
    std::ifstream gt_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> features;

    std::shared_ptr<ilat::fst> lm;

    double cost_scale;

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
        "loss-order2-lat",
        "Calculate loss",
        {
            {"frame-batch", "", false},
            {"lat-batch", "", true},
            {"gt-batch", "", true},
            {"lm", "", true},
            {"param", "", true},
            {"features", "", true},
            {"loss", "", true},
            {"cost-scale", "", false},
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

    cost_scale = 1;
    if (ebt::in(std::string("cost-scale"), args)) {
        cost_scale = std::stod(args.at("cost-scale"));
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 0;

    while (1) {

        ilat::fst lat = ilat::load_lattice(lat_batch, label_id);
        std::vector<segcost::segment<int>> gt_segs = util::load_segments(gt_batch, label_id);

        if (!lat_batch || !gt_batch) {
            break;
        }

        std::vector<std::vector<double>> frames;
        if (ebt::in(std::string("frame-batch"), args)) {
            frames = speech::load_frame_batch(frame_batch);
        }

        std::cout << lat.data->name << std::endl;

        ilat::add_eps_loops(lat);

        ilat::lazy_pair_mode1 composed_fst { lat, *lm };

        fscrf::fscrf_pair_data graph_data;
        graph_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(fst::topo_order(composed_fst));
        graph_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        graph_data.weight_func = fscrf::make_pair_weights(features, var_tree, frames);

        fscrf::loss_func *loss_func;

        if (args.at("loss") == "hinge-loss") {
            loss_func = new fscrf::hinge_loss_pair { graph_data, gt_segs, sils, cost_scale };
        }

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        delete loss_func;

        std::cout << std::endl;

        ++i;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

}

