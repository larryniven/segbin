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
    std::shared_ptr<tensor_tree::vertex> opt_data;

    int layer;
    std::shared_ptr<tensor_tree::vertex> nn_param;
    std::shared_ptr<tensor_tree::vertex> pred_param;
    std::shared_ptr<tensor_tree::vertex> nn_opt_data;
    std::shared_ptr<tensor_tree::vertex> pred_opt_data;

    double step_size;
    double decay;
    double momentum;

    std::vector<std::string> features;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    double cost_scale;

    double dropout;
    double dropout_scale;

    int seed;

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
        "learn-order1-lat",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"lat-batch", "", true},
            {"gt-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", false},
            {"nn-opt-data", "", false},
            {"step-size", "", true},
            {"decay", "", false},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"loss", "", true},
            {"cost-scale", "", false},
            {"dropout", "", false},
            {"dropout-scale", "", false},
            {"seed", "", false},
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

    save_every = std::numeric_limits<int>::max();
    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    }

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    output_nn_param = "nn-param-last";
    if (ebt::in(std::string("output-nn-param"), args)) {
        output_nn_param = args.at("output-nn-param");
    }

    output_nn_opt_data = "nn-opt-data-last";
    if (ebt::in(std::string("output-nn-opt-data"), args)) {
        output_nn_opt_data = args.at("output-nn-opt-data");
    }

    features = ebt::split(args.at("features"), ",");

    param = fscrf::make_tensor_tree(features);
    tensor_tree::load_tensor(param, args.at("param"));
    opt_data = fscrf::make_tensor_tree(features);
    tensor_tree::load_tensor(opt_data, args.at("opt-data"));

    if (ebt::in(std::string("nn-param"), args)) {
        std::tie(layer, nn_param, pred_param)
            = fscrf::load_lstm_param(args.at("nn-param"));
    }

    if (ebt::in(std::string("nn-opt-data"), args)) {
        std::tie(layer, nn_opt_data, pred_opt_data)
            = fscrf::load_lstm_param(args.at("nn-opt-data"));
    }

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

    if (ebt::in(std::string("momentum"), args)) {
        momentum = std::stod(args.at("momentum"));
    }

    label_id = util::load_label_id(args.at("label"));
    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        id_label[p.second] = p.first;
    }

    if (ebt::in(std::string("sils"), args)) {
        for (auto& s: ebt::split(args.at("sils"))) {
            sils.push_back(label_id.at(s));
        }
    }

    cost_scale = 1;
    if (ebt::in(std::string("cost-scale"), args)) {
        cost_scale = std::stod(args.at("cost-scale"));
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    dropout_scale = 0;
    if (ebt::in(std::string("dropout-scale"), args)) {
        dropout_scale = std::stod(args.at("dropout-scale"));
    }

    seed = 0;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 0;

    std::default_random_engine gen { seed };

    while (1) {

        ilat::fst lat = ilat::load_lattice(lat_batch, label_id);
        std::vector<segcost::segment<int>> gt_segs = util::load_segments(gt_batch, label_id);

        if (!lat_batch || !gt_batch) {
            break;
        }

        std::cout << lat.data->name << std::endl;

        std::vector<std::vector<double>> frames;
        if (ebt::in(std::string("frame-batch"), args)) {
            frames = speech::load_frame_batch(frame_batch);
        }

        autodiff::computation_graph comp_graph;

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, nn_param);
            pred_var_tree = make_var_tree(comp_graph, pred_param);
        }

        lstm::stacked_bi_lstm_nn_t nn;
        rnn::pred_nn_t pred_nn;

        fscrf::fscrf_data graph_data;
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(lat));
        graph_data.fst = std::make_shared<ilat::fst>(lat);

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<autodiff::op_t> frame_mat;

        if (ebt::in(std::string("frame-batch"), args)) {
            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
            for (int i = 0; i < frames.size(); ++i) {
                frame_ops.push_back(comp_graph.var(la::vector<double>(frames[i])));
            }

            std::vector<std::shared_ptr<autodiff::op_t>> feat_ops;

            if (ebt::in(std::string("nn-param"), args)) {
                if (ebt::in(std::string("dropout-scale"), args)) {
                    nn = lstm::make_stacked_bi_lstm_nn_with_dropout(comp_graph, lstm_var_tree, frame_ops, lstm::lstm_builder{}, dropout_scale);
                } else if (ebt::in(std::string("dropout"), args)) {
                    nn = lstm::make_stacked_bi_lstm_nn_with_dropout(comp_graph, lstm_var_tree, frame_ops, lstm::lstm_builder{}, gen, dropout);
                } else { 
                    nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, lstm::lstm_builder{});
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

        fscrf::loss_func *loss_func;

        if (args.at("loss") == "hinge-loss") {
            loss_func = new fscrf::hinge_loss { graph_data, gt_segs, sils, cost_scale };
        } else if (args.at("loss") == "hinge-loss-gt") {
            loss_func = new fscrf::hinge_loss_gt { graph_data, gt_segs, sils, cost_scale };
        } else if (args.at("loss") == "log-loss") {
            loss_func = new fscrf::log_loss { graph_data, gt_segs, sils };
        }

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad = fscrf::make_tensor_tree(features);
        std::shared_ptr<tensor_tree::vertex> nn_param_grad;
        std::shared_ptr<tensor_tree::vertex> pred_grad;

        if (ebt::in(std::string("nn-param"), args)) {
            nn_param_grad = lstm::make_stacked_bi_lstm_tensor_tree(layer);
            pred_grad = nn::make_pred_tensor_tree();
        }

        if (ell > 0) {
            loss_func->grad();

            graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            if (ebt::in(std::string("nn-param"), args)) {
                autodiff::grad(frame_mat, autodiff::grad_funcs);
                tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);
                tensor_tree::copy_grad(pred_grad, pred_var_tree);
            }

            double w1 = 0;

            if (ebt::in(std::string("nn-param"), args)) {
                w1 = tensor_tree::get_matrix(nn_param->children[0]->children[0]->children[0])(0, 0);
            }

            if (ebt::in(std::string("decay"), args)) {
                tensor_tree::rmsprop_update(param, param_grad, opt_data,
                    decay, step_size);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::rmsprop_update(nn_param, nn_param_grad, nn_opt_data,
                        decay, step_size);
                    tensor_tree::rmsprop_update(pred_param, pred_grad, pred_opt_data,
                        decay, step_size);
                }
            } else if (ebt::in(std::string("momentum"), args)) {
                tensor_tree::const_step_update_momentum(param, param_grad, opt_data,
                    step_size, momentum);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::const_step_update_momentum(nn_param, nn_param_grad, nn_opt_data,
                        step_size, momentum);
                    tensor_tree::const_step_update_momentum(pred_param, pred_grad, pred_opt_data,
                        step_size, momentum);
                }
            } else {
                tensor_tree::adagrad_update(param, param_grad, opt_data,
                    step_size);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::adagrad_update(nn_param, nn_param_grad, nn_opt_data,
                        step_size);
                    tensor_tree::adagrad_update(pred_param, pred_grad, pred_opt_data,
                        step_size);
                }
            }

            double w2 = 0;

            if (ebt::in(std::string("nn-param"), args)) {
                w2 = tensor_tree::get_matrix(nn_param->children[0]->children[0]->children[0])(0, 0);
                std::cout << "weight: " << w1 << " update: " << w2 - w1 << " ratio: " << (w2 - w1) / w1 << std::endl;
            }
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        delete loss_func;

        std::cout << std::endl;

        ++i;

        if (i % save_every == 0) {
            tensor_tree::save_tensor(param, "param-last");
            tensor_tree::save_tensor(opt_data, "opt-data-last");

            if (ebt::in(std::string("nn-param"), args)) {
                fscrf::save_lstm_param(nn_param, pred_param, "nn-param-last");
                fscrf::save_lstm_param(nn_opt_data, pred_opt_data, "nn-opt-data-last");
            }
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

    tensor_tree::save_tensor(param, output_param);
    tensor_tree::save_tensor(opt_data, output_opt_data);

    if (ebt::in(std::string("nn-param"), args)) {
        fscrf::save_lstm_param(nn_param, pred_param, output_nn_param);
        fscrf::save_lstm_param(nn_opt_data, pred_opt_data, output_nn_opt_data);
    }

}

