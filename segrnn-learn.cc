#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    fscrf::learning_args l_args;

    double dropout;

    double clip;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-learn",
        "Learn segmental RNN",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", true},
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
            {"const-step-update", "", false},
            {"label", "", true},
            {"clip", "", false},
            {"dropout", "", false},
            {"seed", "", false},
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

    label_batch.open(args.at("label-batch"));

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

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    fscrf::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 0;

    while (1) {

        fscrf::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        std::vector<int> label_seq = util::load_label_seq(label_batch, l_args.label_id);

        if (!label_batch) {
            break;
        }

        std::cout << "sample: " << i + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        lstm_var_tree = make_var_tree(comp_graph, l_args.nn_param);
        pred_var_tree = make_var_tree(comp_graph, l_args.pred_param);

        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        std::shared_ptr<lstm::transcriber> trans = fscrf::make_transcriber(l_args);

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops = (*trans)(lstm_var_tree, frame_ops);

        if (ebt::in(std::string("logsoftmax"), args)) {
            pred_nn = rnn::make_pred_nn(pred_var_tree, feat_ops);
            feat_ops = pred_nn.logprob;
        }

        std::cout << "frames: " << s.frames.size() << " downsampled: " << feat_ops.size() << std::endl;

        if (feat_ops.size() < label_seq.size()) {
            continue;
        }

        fscrf::make_graph(s, l_args, feat_ops.size());

        auto frame_mat = autodiff::row_cat(feat_ops);

        autodiff::eval(frame_mat, autodiff::eval_funcs);

        if (dropout == 0.0) {
            s.graph_data.weight_func = fscrf::make_weights(l_args.features, var_tree, frame_mat);
        } else {
            s.graph_data.weight_func = fscrf::make_weights(l_args.features, var_tree, frame_mat,
                dropout, &l_args.gen);
        }

        fscrf::loss_func *loss_func;

        loss_func = new fscrf::marginal_log_loss { s.graph_data, label_seq };

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / label_seq.size() << std::endl;

#if 0
        {
            fscrf::learning_args l_args2 = l_args;
            l_args2.param = tensor_tree::copy_tree(l_args.param);
            l_args2.opt_data = tensor_tree::copy_tree(l_args.opt_data);

            auto& v = tensor_tree::get_vector(l_args2.param->children[0]->children[8]);
            v(0) += 1e-8;

            fscrf::learning_sample s2 { l_args2 };
            s2.frames = s.frames;

            autodiff::computation_graph comp_graph2;
            std::shared_ptr<tensor_tree::vertex> var_tree2
                = tensor_tree::make_var_tree(comp_graph2, l_args2.param);

            auto one2 = comp_graph2.var(one_vec);

            std::shared_ptr<tensor_tree::vertex> lstm_var_tree2;
            std::shared_ptr<tensor_tree::vertex> pred_var_tree2;
            lstm_var_tree2 = make_var_tree(comp_graph2, l_args2.nn_param);
            pred_var_tree2 = make_var_tree(comp_graph2, l_args2.pred_param);

            lstm::stacked_bi_lstm_nn_t nn2;
            rnn::pred_nn_t pred_nn2;

            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops2;
            for (int i = 0; i < s2.frames.size(); ++i) {
                frame_ops2.push_back(comp_graph2.var(la::vector<double>(s2.frames[i])));
            }

            std::vector<std::shared_ptr<autodiff::op_t>> feat_ops2;

            std::shared_ptr<lstm::bi_lstm_builder> builder2
                = std::make_shared<lstm::dyer_bi_lstm_builder>(lstm::dyer_bi_lstm_builder{one2});

            // std::shared_ptr<lstm::bi_lstm_builder> builder2
            //     = std::make_shared<lstm::bi_lstm_builder>(lstm::bi_lstm_builder{});

            if (ebt::in(std::string("dropout"), args)) {
                std::cout << "unable to compute gradient when dropout is on" << std::endl;
                exit(1);
            }

            if (ebt::in(std::string("subsampling"), args)) {
                builder2 = std::make_shared<lstm::bi_lstm_input_subsampling>(
                    lstm::bi_lstm_input_subsampling { builder2 });
            }

            nn2 = lstm::make_stacked_bi_lstm_nn(lstm_var_tree2, frame_ops2, *builder2);
            if (ebt::in(std::string("logsoftmax"), args)) {
                pred_nn2 = rnn::make_pred_nn(pred_var_tree2, nn2.layer.back().output);
                feat_ops2 = pred_nn2.logprob;
            } else {
                feat_ops2 = nn2.layer.back().output;
            }

            fscrf::make_graph(s2, l_args2, feat_ops2.size());

            auto frame_mat2 = autodiff::row_cat(feat_ops2);

            autodiff::eval(frame_mat2, autodiff::eval_funcs);

            s2.graph_data.weight_func = fscrf::make_weights(l_args2.features, var_tree2, frame_mat2);

            fscrf::loss_func *loss_func2;

            loss_func2 = new fscrf::marginal_log_loss { s2.graph_data, label_seq };

            double ell2 = loss_func2->loss();

            std::cout << "numeric grad: " << (ell2 - ell) / 1e-8 << std::endl;

        }
#endif

        std::shared_ptr<tensor_tree::vertex> param_grad = fscrf::make_tensor_tree(l_args.features);
        std::shared_ptr<tensor_tree::vertex> nn_param_grad;
        std::shared_ptr<tensor_tree::vertex> pred_grad;

        nn_param_grad = fscrf::make_lstm_tensor_tree(l_args.outer_layer, l_args.inner_layer);
        pred_grad = nn::make_pred_tensor_tree();

        if (ell > 0) {
            loss_func->grad();

            s.graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            autodiff::grad(frame_mat, autodiff::grad_funcs);
            tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);

            auto& v = tensor_tree::get_vector(param_grad->children[0]->children[8]);
            std::cout << "analytic grad: " << v(0) << std::endl;

            if (ebt::in(std::string("logsoftmax"), args)) {
                tensor_tree::copy_grad(pred_grad, pred_var_tree);
            }

            if (ebt::in(std::string("clip"), args)) {
                double n1 = tensor_tree::norm(nn_param_grad);

                double n2 = 0;
                if (ebt::in(std::string("logsoftmax"), args)) {
                    n2 = tensor_tree::norm(pred_grad);
                }

                double n3 = tensor_tree::norm(param_grad);

                double n = std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);

                if (n > clip) {
                    tensor_tree::imul(nn_param_grad, clip / n);

                    if (ebt::in(std::string("logsoftmax"), args)) {
                        tensor_tree::imul(pred_grad, clip / n);
                    }

                    tensor_tree::imul(param_grad, clip / n);

                    std::cout << "grad norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            std::shared_ptr<tensor_tree::vertex> param_bak = tensor_tree::copy_tree(l_args.param);
            std::shared_ptr<tensor_tree::vertex> opt_data_bak = tensor_tree::copy_tree(l_args.opt_data);

            // double v1 = tensor_tree::get_matrix(l_args.param->children[0])(l_args.label_id.at("sil") - 1, 0);

            double w1 = tensor_tree::get_matrix(l_args.nn_param->children[0]->children[0]->children[0]->children[0])(0, 0);

            if (ebt::in(std::string("decay"), l_args.args)) {
                tensor_tree::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.decay, l_args.step_size);
                tensor_tree::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.decay, l_args.step_size);
                if (ebt::in(std::string("logsoftmax"), args)) {
                    tensor_tree::rmsprop_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                        l_args.decay, l_args.step_size);
                }
            } else if (ebt::in(std::string("momentum"), l_args.args)) {
                tensor_tree::const_step_update_momentum(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size, l_args.momentum);
                tensor_tree::const_step_update_momentum(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.step_size, l_args.momentum);
                if (ebt::in(std::string("logsoftmax"), args)) {
                    tensor_tree::const_step_update_momentum(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                        l_args.step_size, l_args.momentum);
                }
            } else if (ebt::in(std::string("const-step-update"), l_args.args)) {
                tensor_tree::const_step_update(l_args.param, param_grad,
                    l_args.step_size);
                tensor_tree::const_step_update(l_args.nn_param, nn_param_grad,
                    l_args.step_size);
                if (ebt::in(std::string("logsoftmax"), args)) {
                    tensor_tree::const_step_update(l_args.pred_param, pred_grad,
                        l_args.step_size);
                }
            } else {
                tensor_tree::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size);
                tensor_tree::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.step_size);
                if (ebt::in(std::string("logsoftmax"), args)) {
                    tensor_tree::adagrad_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                        l_args.step_size);
                }
            }

            // double v2 = tensor_tree::get_matrix(l_args.param->children[0])(l_args.label_id.at("sil") - 1, 0);
            // std::cout << "weight: " << v1 << " update: " << v2 - v1 << " ratio: " << (v2 - v1) / v1 << std::endl;

            double w2 = tensor_tree::get_matrix(l_args.nn_param->children[0]->children[0]->children[0]->children[0])(0, 0);
            std::cout << "weight: " << w1 << " update: " << w2 - w1 << " ratio: " << (w2 - w1) / w1 << std::endl;
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        double n1 = tensor_tree::norm(l_args.nn_param);

        double n2 = 0;
        if (ebt::in(std::string("logsoftmax"), args)) {
            n2 = tensor_tree::norm(l_args.pred_param);
        }

        double n3 = tensor_tree::norm(l_args.param);

        std::cout << "norm: " << std::sqrt(n1 * n1 + n2 * n2 + n3 * n3) << std::endl;

        std::cout << std::endl;

        ++i;

        if (i % save_every == 0) {
            tensor_tree::save_tensor(l_args.param, "param-last");
            fscrf::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
                l_args.nn_param, l_args.pred_param, "nn-param-last");

            tensor_tree::save_tensor(l_args.opt_data, "opt-data-last");
            fscrf::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
               l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
        }

        delete loss_func;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

    tensor_tree::save_tensor(l_args.param, output_param);
    fscrf::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
        l_args.nn_param, l_args.pred_param, output_nn_param);

    tensor_tree::save_tensor(l_args.opt_data, output_opt_data);
    fscrf::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
        l_args.nn_opt_data, l_args.pred_opt_data, output_nn_opt_data);

}

