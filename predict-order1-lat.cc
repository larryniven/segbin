#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;
    std::ifstream lat_batch;

    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> nn_param;
    std::shared_ptr<tensor_tree::vertex> pred_param;

    int layer;
    double dropout_scale;

    std::vector<std::string> features;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-order1-lat",
        "Predict with segmental CRF",
        {
            {"frame-batch", "", false},
            {"lat-batch", "", true},
            {"param", "", true},
            {"nn-param", "", false},
            {"features", "", true},
            {"label", "", true},
            {"dropout-scale", "", false}
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

    lat_batch.open(args.at("lat-batch"));

    features = ebt::split(args.at("features"), ",");

    param = fscrf::make_tensor_tree(features);
    tensor_tree::load_tensor(param, args.at("param"));

    if (ebt::in(std::string("nn-param"), args)) {
        std::tie(layer, nn_param, pred_param)
            = fscrf::load_lstm_param(args.at("nn-param"));
    }

    dropout_scale = 0;
    if (ebt::in(std::string("dropout-scale"), args)) {
        dropout_scale = std::stod(args.at("dropout-scale"));
    }

    label_id = util::load_label_id(args.at("label"));
    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        id_label[p.second] = p.first;
    }
}

void prediction_env::run()
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

        fscrf::fscrf_data graph_data;
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(lat));
        graph_data.fst = std::make_shared<ilat::fst>(lat);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<autodiff::op_t> frame_mat;

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, nn_param);
            pred_var_tree = make_var_tree(comp_graph, pred_param);
        }

        lstm::stacked_bi_lstm_nn_t nn;
        rnn::pred_nn_t pred_nn;

        if (ebt::in(std::string("frame-batch"), args)) {
            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
            for (int i = 0; i < frames.size(); ++i) {
                frame_ops.push_back(comp_graph.var(la::vector<double>(frames[i])));
            }

            std::vector<std::shared_ptr<autodiff::op_t>> feat_ops;

            if (ebt::in(std::string("nn-param"), args)) {
                if (ebt::in(std::string("dropout-scale"), args)) {
                    lstm::bi_lstm_input_scaling builder { dropout_scale,
                        std::make_shared<lstm::bi_lstm_builder>(lstm::bi_lstm_builder{}) };
                    nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, builder);
                } else { 
                    nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, lstm::bi_lstm_builder{});
                }
                pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);
                feat_ops = pred_nn.logprob;
            } else {
                feat_ops = frame_ops;
            }

            frame_mat = autodiff::row_cat(feat_ops);

            autodiff::eval(frame_mat, autodiff::eval_funcs);
            graph_data.weight_func = fscrf::make_weights(features, var_tree, frame_mat);
        } else {
            graph_data.weight_func = fscrf::make_lat_weights(features, var_tree);
        }

        fscrf::fscrf_fst scrf { graph_data };

        fst::forward_one_best<fscrf::fscrf_fst> one_best;

        for (auto& i: scrf.initials()) {
            one_best.extra[i] = fst::forward_one_best<fscrf::fscrf_fst>::extra_data {-1, 0};
        }

        one_best.merge(scrf, *graph_data.topo_order);

        std::vector<int> path = one_best.best_path(scrf);

        for (auto& e: path) {
            std::cout << id_label.at(scrf.output(e)) << " ";
        }
        std::cout << "(" << lat.data->name << ")" << std::endl;

        ++i;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

}

