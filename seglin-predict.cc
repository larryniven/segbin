#include "seg/seg-util.h"
#include "util/speech.h"
#include "util/util.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"

struct prediction_env {

    std::vector<std::string> features;

    std::ifstream frame_batch;

    int max_seg;
    int min_seg;
    int stride;

    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "seglin-predict",
        "Decode with a linear segmental model",
        {
            {"frame-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
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

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    features = ebt::split(args.at("features"), ",");

    param = seg::make_tensor_tree(features);
    tensor_tree::load_tensor(param, args.at("param"));

    frame_batch.open(args.at("frame-batch"));

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    stride = 1;
    if (ebt::in(std::string("stride"), args)) {
        stride = std::stoi(args.at("stride"));
    }

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void prediction_env::run()
{
    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<double> frame_cat;
        frame_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            frame_cat.insert(frame_cat.end(), frames[i].begin(), frames[i].end());
        }

        unsigned int nframes = frames.size();
        unsigned int ndim = frames.front().size();

        std::shared_ptr<autodiff::op_t> frame_mat = comp_graph.var(la::cpu::weak_tensor<double>(
            frame_cat.data(), { nframes, ndim }));

        seg::iseg_data graph_data;
        graph_data.fst = seg::make_graph(frames.size(), label_id, id_label, min_seg, max_seg, stride);
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(*graph_data.fst));

        graph_data.weight_func = seg::make_weights(features, var_tree, frame_mat);

        seg::seg_fst<seg::iseg_data> graph { graph_data };

        fst::forward_one_best<seg::seg_fst<seg::iseg_data>> one_best;
        for (auto& i: graph.initials()) {
            one_best.extra[i] = {-1, 0};
        }
        one_best.merge(graph, *graph_data.topo_order);

        std::vector<int> path = one_best.best_path(graph);

        for (int e: path) {
            std::cout << id_label.at(graph.output(e)) << " ";
        }
        std::cout << "(" << nsample << ".dot)" << std::endl;

        ++nsample;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

}

