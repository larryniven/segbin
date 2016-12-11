#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "nn/lstm-tensor-tree.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gt_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    fscrf::learning_args l_args;

    int subsample_gt_freq;
    double dropout_scale;

    double dropout;

    double clip;

    int mini_batch;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-first",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"gt-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
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
            {"label", "", true},
            {"subsample-gt-freq", "", false},
            {"adam-beta1", "", false},
            {"adam-beta2", "", false},
            {"dropout-scale", "", false},
            {"clip", "", false},
            {"dropout", "", false},
            {"freeze-encoder", "", false},
            {"mini-batch", "", false},
            {"edge-drop", "", false},
            {"seed", "", false},
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
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    gt_batch.open(args.at("gt-batch"));

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

    dropout_scale = 0;
    if (ebt::in(std::string("dropout-scale"), args)) {
        dropout_scale = std::stod(args.at("dropout-scale"));
    }

    subsample_gt_freq = 1;
    if (ebt::in(std::string("subsample-gt-freq"), args)) {
        subsample_gt_freq = std::stoi(args.at("subsample-gt-freq"));
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    mini_batch = 1;
    if (ebt::in(std::string("mini-batch"), args)) {
        mini_batch = std::stoi(args.at("mini-batch"));
    }

    fscrf::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 0;

    std::shared_ptr<tensor_tree::vertex> accu_param_grad
        = fscrf::make_tensor_tree(l_args.features);
    std::shared_ptr<tensor_tree::vertex> accu_nn_param_grad;
    std::shared_ptr<tensor_tree::vertex> accu_pred_grad;

    tensor_tree::resize_as(accu_param_grad, l_args.param);

    if (ebt::in(std::string("nn-param"), args)) {
        accu_nn_param_grad = lstm::make_stacked_bi_lstm_tensor_tree(l_args.outer_layer);
        accu_pred_grad = nn::make_pred_tensor_tree();

        tensor_tree::resize_as(accu_nn_param_grad, l_args.nn_param);
        tensor_tree::resize_as(accu_pred_grad, l_args.pred_param);
    }

    while (1) {

        fscrf::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        s.gt_segs = util::load_segments(gt_batch, l_args.label_id, subsample_gt_freq);

        if (!gt_batch) {
            break;
        }

        std::cout << "sample: " << i + 1 << std::endl;

        fscrf::make_graph(s, l_args);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, l_args.nn_param);
            pred_var_tree = make_var_tree(comp_graph, l_args.pred_param);
        }

        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops;

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans = fscrf::make_transcriber(l_args);
            feat_ops = (*trans)(lstm_var_tree, frame_ops);
            pred_nn = rnn::make_pred_nn(pred_var_tree, feat_ops);
            feat_ops = pred_nn.logprob;
        } else {
            feat_ops = frame_ops;
        }

        auto frame_mat = autodiff::row_cat(feat_ops);

        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = fscrf::make_weights(l_args.features, var_tree, frame_mat);

        fscrf::loss_func *loss_func;
        if (args.at("loss") == "hinge-loss") {
            loss_func = new fscrf::hinge_loss { s.graph_data, s.gt_segs, l_args.sils, l_args.cost_scale };
        } else if (args.at("loss") == "hinge-loss-gt") {
            loss_func = new fscrf::hinge_loss_gt { s.graph_data, s.gt_segs, l_args.sils, l_args.cost_scale };
        } else if (args.at("loss") == "log-loss") {
            loss_func = new fscrf::log_loss { s.graph_data, s.gt_segs, l_args.sils };
        } else if (args.at("loss") == "latent-hinge") {
            std::vector<int> label_seq;

            for (int i = 0; i < s.gt_segs.size(); ++i) {
                label_seq.push_back(s.gt_segs[i].label);
            }

            loss_func = new fscrf::latent_hinge_loss { s.graph_data, label_seq, l_args.sils, l_args.cost_scale };
        } else if (args.at("loss") == "marginal-log-loss") {
            std::vector<int> label_seq;

            for (int i = 0; i < s.gt_segs.size(); ++i) {
                label_seq.push_back(s.gt_segs[i].label);
            }

            loss_func = new fscrf::marginal_log_loss { s.graph_data, label_seq };
        }

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

#if 0
        {
            fscrf::learning_args l_args2 = l_args;
            l_args2.param = tensor_tree::copy_tree(l_args.param);
            l_args2.opt_data = tensor_tree::copy_tree(l_args.opt_data);

            auto& m = tensor_tree::get_matrix(l_args2.param->children[2]);
            m(l_args.label_id.at("sil") - 1, 0) += 1e-8;

            fscrf::learning_sample s2 { l_args2 };
            s2.frames = s.frames;
            s2.gt_segs = s.gt_segs;

            fscrf::make_graph(s2, l_args);

            autodiff::computation_graph comp_graph2;
            std::shared_ptr<tensor_tree::vertex> var_tree2
                = tensor_tree::make_var_tree(comp_graph2, l_args2.param);

            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops2;
            for (int i = 0; i < s2.frames.size(); ++i) {
                frame_ops2.push_back(comp_graph2.var(la::vector<double>(s2.frames[i])));
            }

            auto frame_mat2 = autodiff::row_cat(frame_ops2);

            s2.graph_data.weight_func = fscrf::make_weights(l_args2.features, var_tree2, frame_mat2);

            fscrf::hinge_loss loss_func2 { s2.graph_data, s2.gt_segs, l_args2.sils, l_args2.cost_scale };

            double ell2 = loss_func2.loss();

            std::cout << "numeric grad: " << (ell2 - ell) / 1e-8 << std::endl;

        }
#endif

        std::shared_ptr<tensor_tree::vertex> param_grad = fscrf::make_tensor_tree(l_args.features);
        std::shared_ptr<tensor_tree::vertex> nn_param_grad;
        std::shared_ptr<tensor_tree::vertex> pred_grad;

        if (ebt::in(std::string("nn-param"), args)) {
            nn_param_grad = fscrf::make_lstm_tensor_tree(l_args.outer_layer, -1);
            pred_grad = nn::make_pred_tensor_tree();
        }

        if (ell > 0) {
            loss_func->grad();

            s.graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            auto& m = tensor_tree::get_matrix(param_grad->children[0]);

            std::cout << "analytic grad: " << m(l_args.label_id.at("sil") - 1, 0) << std::endl;

            if (ebt::in(std::string("nn-param"), args)) {
                autodiff::grad(frame_mat, autodiff::grad_funcs);
                tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);
                tensor_tree::copy_grad(pred_grad, pred_var_tree);
            }

            if (ebt::in(std::string("clip"), args)) {
                double n1 = tensor_tree::norm(nn_param_grad);
                double n2 = tensor_tree::norm(pred_grad);
                double n3 = tensor_tree::norm(param_grad);

                double n = std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);

                if (n > clip) {
                    tensor_tree::imul(nn_param_grad, clip / n);
                    tensor_tree::imul(pred_grad, clip / n);
                    tensor_tree::imul(param_grad, clip / n);

                    std::cout << "grad norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            tensor_tree::iadd(accu_param_grad, param_grad);
            if (ebt::in(std::string("nn-param"), args)) {
                tensor_tree::iadd(accu_nn_param_grad, nn_param_grad);
                tensor_tree::iadd(accu_pred_grad, pred_grad);
            }

            if ((i + 1) % mini_batch == 0) {

                tensor_tree::imul(accu_param_grad, 1.0 / mini_batch);
                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::imul(accu_nn_param_grad, 1.0 / mini_batch);
                    tensor_tree::imul(accu_pred_grad, 1.0 / mini_batch);
                }

                std::shared_ptr<tensor_tree::vertex> param_bak = tensor_tree::copy_tree(l_args.param);
                std::shared_ptr<tensor_tree::vertex> opt_data_bak = tensor_tree::copy_tree(l_args.opt_data);

                double v1 = tensor_tree::get_matrix(l_args.param->children[0])(l_args.label_id.at("sil") - 1, 0);
                double w1 = 0;

                if (ebt::in(std::string("nn-param"), args)) {
                    w1 = tensor_tree::get_matrix(l_args.nn_param->children[0]->children[0]->children[0])(0, 0);
                }

                if (ebt::in(std::string("decay"), l_args.args)) {
                    tensor_tree::rmsprop_update(l_args.param, accu_param_grad, l_args.opt_data,
                        l_args.decay, l_args.step_size);

                    if (ebt::in(std::string("nn-param"), l_args.args)) {
                        tensor_tree::rmsprop_update(l_args.nn_param, accu_nn_param_grad, l_args.nn_opt_data,
                            l_args.decay, l_args.step_size);
                        tensor_tree::rmsprop_update(l_args.pred_param, accu_pred_grad, l_args.pred_opt_data,
                            l_args.decay, l_args.step_size);
                    }
                } else if (ebt::in(std::string("adam-beta1"), l_args.args)) {
                    tensor_tree::adam_update(l_args.param, accu_param_grad, l_args.first_moment, l_args.second_moment,
                        l_args.time, l_args.step_size, l_args.adam_beta1, l_args.adam_beta2);

                    if (ebt::in(std::string("nn-param"), l_args.args)) {
                        tensor_tree::adam_update(l_args.nn_param, accu_nn_param_grad, l_args.nn_first_moment, l_args.nn_second_moment,
                            l_args.time, l_args.step_size, l_args.adam_beta1, l_args.adam_beta2);
                        tensor_tree::adam_update(l_args.pred_param, accu_pred_grad, l_args.pred_first_moment, l_args.pred_second_moment,
                            l_args.time, l_args.step_size, l_args.adam_beta1, l_args.adam_beta2);
                    }
                } else if (ebt::in(std::string("momentum"), l_args.args)) {
                    tensor_tree::const_step_update_momentum(l_args.param, accu_param_grad, l_args.opt_data,
                        l_args.step_size, l_args.momentum);

                    if (ebt::in(std::string("nn-param"), l_args.args)) {
                        tensor_tree::const_step_update_momentum(l_args.nn_param, accu_nn_param_grad, l_args.nn_opt_data,
                            l_args.step_size, l_args.momentum);
                        tensor_tree::const_step_update_momentum(l_args.pred_param, accu_pred_grad, l_args.pred_opt_data,
                            l_args.step_size, l_args.momentum);
                    }
                } else {
                    tensor_tree::adagrad_update(l_args.param, accu_param_grad, l_args.opt_data,
                        l_args.step_size);

                    if (ebt::in(std::string("nn-param"), l_args.args)) {
                        tensor_tree::adagrad_update(l_args.nn_param, accu_nn_param_grad, l_args.nn_opt_data,
                            l_args.step_size);
                        tensor_tree::adagrad_update(l_args.pred_param, accu_pred_grad, l_args.pred_opt_data,
                            l_args.step_size);
                    }
                }

                double v2 = tensor_tree::get_matrix(l_args.param->children[0])(l_args.label_id.at("sil") - 1, 0);
                double w2 = 0;

                if (ebt::in(std::string("nn-param"), args)) {
                    w2 = tensor_tree::get_matrix(l_args.nn_param->children[0]->children[0]->children[0])(0, 0);
                }

                std::cout << "weight: " << v1 << " update: " << v2 - v1 << " ratio: " << (v2 - v1) / v1 << std::endl;

                if (ebt::in(std::string("nn-param"), args)) {
                    std::cout << "weight: " << w1 << " update: " << w2 - w1 << " ratio: " << (w2 - w1) / w1 << std::endl;
                }

                if (tensor_tree::has_nan(l_args.opt_data)) {

                    if (tensor_tree::has_nan(accu_param_grad)) {
                        std::cout << "grad has nan" << std::endl;
                    }

                    std::ofstream ofs;
                    ofs.open("param-debug");
                    tensor_tree::save_tensor(param_bak, ofs);
                    ofs.close();

                    ofs.open("opt-data-debug");
                    tensor_tree::save_tensor(opt_data_bak, ofs);
                    ofs.close();

                    exit(1);
                }

                tensor_tree::zero(accu_param_grad);
                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::zero(accu_nn_param_grad);
                    tensor_tree::zero(accu_pred_grad);
                }

            }

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << "gold segs: " << s.gt_segs.size()
            << " frames: " << s.frames.size() << std::endl;

        std::cout << std::endl;

        ++i;
        ++l_args.time;

        if (i % save_every == 0) {
            tensor_tree::save_tensor(l_args.param, "param-last");

            if (ebt::in(std::string("nn-param"), args)) {
                fscrf::save_lstm_param(l_args.nn_param, l_args.pred_param, "nn-param-last");
            }

            if (ebt::in(std::string("adam-beta1"), args)) {
                std::ofstream ofs { "opt-last" };
                ofs << l_args.time << std::endl;
                tensor_tree::save_tensor(l_args.first_moment, ofs);
                tensor_tree::save_tensor(l_args.second_moment, ofs);
            } else {
                tensor_tree::save_tensor(l_args.opt_data, "opt-data-last");

                if (ebt::in(std::string("nn-param"), args)) {
                    fscrf::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
                }
            }
        }

        delete loss_func;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

    tensor_tree::save_tensor(l_args.param, output_param);

    if (ebt::in(std::string("nn-param"), args)) {
        fscrf::save_lstm_param(l_args.nn_param, l_args.pred_param, output_nn_param);
    }

    if (ebt::in(std::string("adam-beta1"), args)) {
        std::ofstream ofs { output_opt_data };
        ofs << l_args.time << std::endl;
        tensor_tree::save_tensor(l_args.first_moment, ofs);
        tensor_tree::save_tensor(l_args.second_moment, ofs);
    } else {
        tensor_tree::save_tensor(l_args.opt_data, output_opt_data);

        if (ebt::in(std::string("nn-param"), args)) {
            fscrf::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, output_nn_opt_data);
        }
    }

}

