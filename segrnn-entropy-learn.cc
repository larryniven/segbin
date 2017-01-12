#include "seg/seg-util.h"
#include "speech/speech.h"
#include "fst/fst-algo.h"
#include "seg/loss.h"
#include <fstream>

struct learning_env {

    speech::batch_indices frame_batch;

    seg::learning_args l_args;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    double dropout;

    double clip;

    int seed;

    std::default_random_engine gen;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-entropy-learn",
        "Train segmental RNN by minimizing entropy",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", false},
            {"nn-opt-data", "", false},
            {"features", "", true},
            {"output-param", "", true},
            {"output-opt-data", "", true},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"step-size", "", true},
            {"const-step-update", "", false},
            {"clip", "", false},
            {"label", "", true},
            {"subsampling", "", false},
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"save-every", "", false}
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
    frame_batch.open(args.at("frame-batch"));

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

    seg::parse_learning_args(l_args, args);

    if (ebt::in(std::string("shuffle"), args)) {
        std::vector<int> indices;
        indices.resize(frame_batch.pos.size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), l_args.gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            frame_batch.pos[i] = pos[indices[i]];
        }
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        seg::learning_sample s { l_args };

        std::cout << "sample: " << nsample << std::endl;

        s.frames = speech::load_frame_batch(frame_batch.at(nsample));

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;

        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, l_args.nn_param);
        }

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::tensor<double>(
                la::vector<double>(s.frames[i]))));
        }

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans
                = seg::make_transcriber(l_args);

            frame_ops = (*trans)(lstm_var_tree->children[0], frame_ops);
        }

        std::cout << "frames: " << s.frames.size() << " subsampled: " << frame_ops.size() << std::endl;

        seg::make_graph(s, l_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = seg::make_weights(l_args.features, var_tree, frame_mat);

        seg::entropy_loss loss { s.graph_data };

        double ell = loss.loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "nats per frame: " << ell / s.frames.size() << std::endl;

#if 0
        {
            seg::learning_args l_args2 = l_args;

            l_args2.param = tensor_tree::copy_tree(l_args.param);

            auto vars = tensor_tree::leaves_pre_order(l_args2.param);
            tensor_tree::get_tensor(l_args2.param->children[0]).data()[0] += 1e-8;

            seg::learning_sample s2 { l_args2 };

            s2.frames = s.frames;

            autodiff::computation_graph comp_graph2;
            std::shared_ptr<tensor_tree::vertex> var_tree2
                = tensor_tree::make_var_tree(comp_graph2, l_args2.param);

            std::shared_ptr<tensor_tree::vertex> lstm_var_tree2;

            if (ebt::in(std::string("nn-param"), args)) {
                lstm_var_tree2 = make_var_tree(comp_graph2, l_args2.nn_param);
            }

            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops2;
            for (int i = 0; i < s2.frames.size(); ++i) {
                frame_ops2.push_back(comp_graph2.var(la::tensor<double>(
                    la::vector<double>(s2.frames[i]))));
            }

            if (ebt::in(std::string("nn-param"), args)) {
                std::shared_ptr<lstm::transcriber> trans2
                    = seg::make_transcriber(l_args2);

                frame_ops2 = (*trans2)(lstm_var_tree2->children[0], frame_ops2);
            }

            seg::make_graph(s2, l_args2, frame_ops2.size());

            auto frame_mat2 = autodiff::row_cat(frame_ops2);
            autodiff::eval(frame_mat2, autodiff::eval_funcs);

            s2.graph_data.weight_func = seg::make_weights(l_args2.features, var_tree2, frame_mat2);

            seg::entropy_loss loss2 { s2.graph_data };

            std::cout << "numeric grad: " << (loss2.loss() - loss.loss()) / 1e-8 << std::endl;
        }
#endif

        if (ell > 0) {
            loss.grad();

            s.graph_data.weight_func->grad();

            autodiff::grad(frame_mat, autodiff::grad_funcs);

            std::shared_ptr<tensor_tree::vertex> param_grad
                = seg::make_tensor_tree(l_args.features);

            std::shared_ptr<tensor_tree::vertex> nn_param_grad;

            if (ebt::in(std::string("nn-param"), args)) {
                nn_param_grad = seg::make_lstm_tensor_tree(l_args.outer_layer, l_args.inner_layer);
            }

            tensor_tree::copy_grad(param_grad, var_tree);

            if (ebt::in(std::string("nn-param"), args)) {
                autodiff::grad(frame_mat, autodiff::grad_funcs);
                tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);

                auto vars = tensor_tree::leaves_pre_order(nn_param_grad);
                std::cout << "analytic grad: " << tensor_tree::get_tensor(vars[0]).data()[0]
                    << std::endl;
            } else {
                auto vars = tensor_tree::leaves_pre_order(param_grad);
                std::cout << "analytic grad: " << tensor_tree::get_tensor(param_grad->children[0]).data()[0]
                    << std::endl;
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars;

            if (ebt::in(std::string("nn-param"), args)) {
                vars = tensor_tree::leaves_pre_order(l_args.nn_param);
            } else {
                vars = tensor_tree::leaves_pre_order(l_args.param);
            }

            double v1 = tensor_tree::get_tensor(vars[0]).data()[0];

            if (ebt::in(std::string("clip"), args)) {
                double n1 = 0;

                if (ebt::in(std::string("nn-param"), args)) {
                    n1 = tensor_tree::norm(nn_param_grad);
                }

                double n2 = tensor_tree::norm(param_grad);

                double n = std::sqrt(n1 * n1 + n2 * n2);

                if (n > clip) {
                    if (ebt::in(std::string("nn-param"), args)) {
                        tensor_tree::imul(nn_param_grad, clip / n);
                    }

                    tensor_tree::imul(param_grad, clip / n);

                    std::cout << "grad norm: " << n
                        << " clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            if (ebt::in(std::string("decay"), l_args.args)) {
                tensor_tree::rmsprop_update(l_args.param, param_grad,
                    l_args.opt_data, l_args.decay, l_args.step_size);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::rmsprop_update(l_args.nn_param, nn_param_grad,
                        l_args.nn_opt_data, l_args.decay, l_args.step_size);
                }
            } else if (ebt::in(std::string("momentum"), l_args.args)) {
                tensor_tree::const_step_update_momentum(l_args.param, param_grad,
                    l_args.opt_data, l_args.step_size, l_args.momentum);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::const_step_update_momentum(l_args.nn_param, nn_param_grad,
                        l_args.nn_opt_data, l_args.step_size, l_args.momentum);
                }
            } else if (ebt::in(std::string("const-step-update"), l_args.args)) {
                tensor_tree::const_step_update(l_args.param, param_grad,
                    l_args.step_size);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::const_step_update(l_args.nn_param, nn_param_grad,
                        l_args.step_size);
                }
            } else {
                tensor_tree::adagrad_update(l_args.param, param_grad,
                    l_args.opt_data, l_args.step_size);

                if (ebt::in(std::string("nn-param"), args)) {
                    tensor_tree::adagrad_update(l_args.nn_param, nn_param_grad,
                        l_args.nn_opt_data, l_args.step_size);
                }
            }

            double v2 = tensor_tree::get_tensor(vars[0]).data()[0];

            std::cout << "weight: " << v1 << " update: " << v2 - v1
                << " ratio: " << (v2 - v1) / v1 << std::endl;

        } else {
            std::cout << "loss is less than or equal to zero.  skipping." << std::endl;
        }

        double n1 = 0;

        if (ebt::in(std::string("nn-param"), args)) {
            n1 = tensor_tree::norm(l_args.nn_param);
        }

        double n2 = tensor_tree::norm(l_args.param);

        std::cout << "norm: " << std::sqrt(n1 * n1 + n2 * n2) << std::endl;

        std::cout << std::endl;

        if (ebt::in(std::string("save-every"), args)) {
            if (nsample == std::stoi(args.at("save-every"))) {
                tensor_tree::save_tensor(l_args.param, "param-last");
                tensor_tree::save_tensor(l_args.opt_data, "opt-data-last");

                if (ebt::in(std::string("nn-param"), args)) {
                    seg::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
                        l_args.nn_param, "nn-param-last");
                    seg::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
                        l_args.nn_opt_data, "nn-opt-data-last");
                }
            }
        }

        ++nsample;
    }

    tensor_tree::save_tensor(l_args.param, output_param);
    tensor_tree::save_tensor(l_args.opt_data, output_opt_data);

    if (ebt::in(std::string("nn-param"), args)) {
        seg::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
            l_args.nn_param, output_nn_param);
        seg::save_lstm_param(l_args.outer_layer, l_args.inner_layer,
            l_args.nn_opt_data, output_nn_opt_data);
    }

}

