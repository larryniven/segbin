#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct loss_env {

    std::ifstream frame_batch;
    std::ifstream gt_batch;

    fscrf::learning_args l_args;

    int subsample_gt_freq;

    std::unordered_map<std::string, std::string> args;

    loss_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "loss-order1-full",
        "Compute loss",
        {
            {"frame-batch", "", false},
            {"gt-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"loss", "", true},
            {"cost-scale", "", false},
            {"label", "", true},
            {"subsample-gt-freq", "", false}
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

    loss_env env { args };

    env.run();

    return 0;
}

loss_env::loss_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    gt_batch.open(args.at("gt-batch"));

    subsample_gt_freq = 1;
    if (ebt::in(std::string("subsample-gt-freq"), args)) {
        subsample_gt_freq = std::stoi(args.at("subsample-gt-freq"));
    }

    fscrf::parse_learning_args(l_args, args);
}

void loss_env::run()
{
    ebt::Timer timer;

    int i = 1;

    while (1) {

        fscrf::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        s.gt_segs = util::load_segments(gt_batch, l_args.label_id, subsample_gt_freq);

        if (!gt_batch) {
            break;
        }

        fscrf::make_graph(s, l_args);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        auto frame_mat = autodiff::col_cat(frame_ops);

        s.graph_data.weight_func = fscrf::make_weights(l_args.features, var_tree, frame_mat);

        fscrf::hinge_loss loss_func { s.graph_data, s.gt_segs, l_args.sils, l_args.cost_scale };

        double ell = loss_func.loss();

        std::cout << "loss: " << ell << std::endl;

        std::cout << "gold segs: " << s.gt_segs.size()
            << " frames: " << s.frames.size() << std::endl;

        std::cout << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

}

