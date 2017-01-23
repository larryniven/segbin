#include "seg/seg-util.h"
#include "speech/speech.h"
#include "fst/fst-algo.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    int inner_layer;
    int outer_layer;
    std::shared_ptr<tensor_tree::vertex> nn_param;

    seg::inference_args i_args;

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
            {"stride", "", false},
            {"param", "", true},
            {"nn-param", "", false},
            {"features", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"logsoftmax", "", false},
            {"print-path", "", false},
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

    if (ebt::in(std::string("nn-param"), args)) {
        std::tie(outer_layer, inner_layer, nn_param)
            = seg::load_lstm_param(args.at("nn-param"));
    }

    seg::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int nsample = 1;

    while (1) {

        seg::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;

        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, nn_param);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::tensor<double>(
                la::vector<double>(s.frames[i]))));
        }

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans
                = seg::make_transcriber(outer_layer, inner_layer, args, nullptr);

            if (ebt::in(std::string("logsoftmax"), args)) {
                trans = std::make_shared<lstm::logsoftmax_transcriber>(
                    lstm::logsoftmax_transcriber { trans });
                frame_ops = (*trans)(lstm_var_tree, frame_ops);
            } else {
                frame_ops = (*trans)(lstm_var_tree->children[0], frame_ops);
            }
        }

        seg::make_graph(s, i_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = seg::make_weights(i_args.features, var_tree, frame_mat);

        seg::seg_fst<seg::iseg_data> graph { s.graph_data };

        std::vector<int> path = fst::shortest_path(graph, *s.graph_data.topo_order);

        if (ebt::in(std::string("print-path"), args)) {
            std::cout << nsample << ".txt" << std::endl;
            for (auto& e: path) {
                std::cout << graph.time(graph.tail(e)) << " " << graph.time(graph.head(e)) << " " << i_args.id_label.at(graph.output(e)) << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            for (auto& e: path) {
                std::cout << i_args.id_label.at(graph.output(e)) << " ";
            }
            std::cout << "(" << nsample << ".dot)";
            std::cout << std::endl;
        }

        ++nsample;
    }
}

