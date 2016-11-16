#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    fscrf::inference_args i_args;

    double dropout;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-first",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"dropout", "", false},
            {"subsampling", "", false},
            {"logsoftmax", "", false}
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

    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    fscrf::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int i = 1;

    la::vector<double> one_vec;
    auto& m = tensor_tree::get_matrix(i_args.nn_param->children[0]->children[0]->children[0]);
    one_vec.resize(m.rows(), 1);

    while (1) {

        fscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        lstm_var_tree = make_var_tree(comp_graph, i_args.nn_param);
        pred_var_tree = make_var_tree(comp_graph, i_args.pred_param);

        lstm::stacked_bi_lstm_nn_t nn;
        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        std::shared_ptr<autodiff::op_t> one = comp_graph.var(one_vec);

        std::shared_ptr<lstm::bi_lstm_builder> builder
            = std::make_shared<lstm::dyer_bi_lstm_builder>(lstm::dyer_bi_lstm_builder{one});

        if (ebt::in(std::string("dropout"), args)) {
            builder = std::make_shared<lstm::bi_lstm_input_scaling>(
                lstm::bi_lstm_input_scaling { 1.0 - dropout, builder });
        }

        if (ebt::in(std::string("subsampling"), args)) {
            builder = std::make_shared<lstm::bi_lstm_input_subsampling>(
                lstm::bi_lstm_input_subsampling { builder });
        }

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops;

        nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, *builder);

        if (ebt::in(std::string("logsoftmax"), args)) {
            pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);
            feat_ops = pred_nn.logprob;
        } else {
            feat_ops = nn.layer.back().output;
        }

        fscrf::make_graph(s, i_args, feat_ops.size());

        auto frame_mat = autodiff::row_cat(feat_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = fscrf::make_weights(i_args.features, var_tree, frame_mat);

        fscrf::fscrf_data graph_path_data;
        graph_path_data.fst = scrf::shortest_path(s.graph_data);

        fscrf::fscrf_fst graph_path { graph_path_data };

        fscrf::fscrf_fst graph { s.graph_data };

        for (auto& e: graph_path.edges()) {
            std::cout << i_args.id_label.at(graph_path.output(e)) << " ";
        }
        std::cout << "(" << i << ".dot)";
        std::cout << std::endl;

        ++i;
    }
}

