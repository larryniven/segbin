#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    fscrf::inference_args i_args;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-predict",
        "Predict with segmental RNN",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", false},
            {"features", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"logsoftmax", "", false},
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
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    fscrf::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int nsample = 1;

    while (1) {

        fscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = make_var_tree(comp_graph, i_args.nn_param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::tensor<double>(
                la::vector<double>(s.frames[i]))));
        }

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans
                = fscrf::make_transcriber(i_args);

            if (ebt::in(std::string("logsoftmax"), args)) {
                trans = std::make_shared<lstm::logsoftmax_transcriber>(
                    lstm::logsoftmax_transcriber { trans });
                frame_ops = (*trans)(lstm_var_tree, frame_ops);
            } else {
                frame_ops = (*trans)(lstm_var_tree->children[0], frame_ops);
            }
        }

        fscrf::make_graph(s, i_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = fscrf::make_weights(i_args.features, var_tree, frame_mat);

        fscrf::fscrf_data graph_path_data;
        graph_path_data.fst = scrf::shortest_path(s.graph_data);

        fscrf::fscrf_fst graph_path { graph_path_data };

        fscrf::fscrf_fst graph { s.graph_data };

        for (auto& e: graph_path.edges()) {
            std::cout << i_args.id_label.at(graph_path.output(e)) << " ";
        }
        std::cout << "(" << nsample << ".dot)";
        std::cout << std::endl;

        ++nsample;
    }
}

