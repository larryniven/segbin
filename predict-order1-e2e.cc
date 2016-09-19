#include "seg/iscrf_e2e.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "autodiff/autodiff.h"
#include "nn/lstm.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    std::ifstream lattice_batch;

    iscrf::e2e::inference_args i_args;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-order1-e2e",
        "Decode with segmental CRF",
        {
            {"frame-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"dropout", "", false},
            {"frame-softmax", "", false}
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

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
    }

    iscrf::e2e::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int i = 1;

    while (1) {

        iscrf::sample s { i_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (auto& f: frames) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(f)));
        }

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.nn_param);
        std::shared_ptr<tensor_tree::vertex> pred_var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.pred_param);

        lstm::stacked_bi_lstm_nn_t nn;

        if (ebt::in(std::string("dropout"), args)) {
            nn = lstm::make_stacked_bi_lstm_nn_with_dropout(comp_graph, lstm_var_tree, frame_ops, lstm::lstm_builder{}, i_args.dropout);
        } else {
            nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, lstm::lstm_builder{});
        }

        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops;

        if (ebt::in(std::string("frame-softmax"), args)) {
            pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

            feat_ops = pred_nn.logprob;
        } else {
            feat_ops = nn.layer.back().output;
        }

        auto order = autodiff::topo_order(feat_ops);
        autodiff::eval(order, autodiff::eval_funcs);

        std::vector<std::vector<double>> feats;
        for (auto& o: feat_ops) {
            auto& f = autodiff::get_output<la::vector<double>>(o);
            feats.push_back(std::vector<double> {f.data(), f.data() + f.size()});
        }

        s.frames = feats;

        if (ebt::in(std::string("lattice-batch"), args)) {
            ilat::fst lat = ilat::load_lattice(lattice_batch, i_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            iscrf::make_lattice(lat, s, i_args);
        } else {
            iscrf::make_graph(s, i_args);
        }

        iscrf::parameterize(s.graph_data, s.graph_alloc, s.frames, i_args);

        std::shared_ptr<ilat::fst> graph_path = scrf::shortest_path<iscrf::iscrf_data>(s.graph_data);

        for (auto& e: graph_path->edges()) {
            std::cout << i_args.id_label.at(graph_path->output(e)) << " ";
        }
        std::cout << "(" << i << ".phn)" << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

}

