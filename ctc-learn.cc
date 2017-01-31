#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::string output_param;
    std::string output_opt_data;

    seg::learning_args l_args;

    int inner_layer;
    int outer_layer;
    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    double dropout;

    double clip;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "ctc-learn",
        "Train RNN with CTC",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"momentum", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"const-step-update", "", false},
            {"label", "", true},
            {"clip", "", false},
            {"dropout", "", false},
            {"seed", "", false},
            {"subsampling", "", false},
            {"output-dropout", "", false},
            {"shuffle", "", false}
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
    label_batch.open(args.at("label-batch"));

    std::tie(outer_layer, inner_layer, param)
        = seg::load_lstm_param(args.at("param"));

    std::tie(outer_layer, inner_layer, opt_data)
        = seg::load_lstm_param(args.at("opt-data"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
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

        pos = label_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            label_batch.pos[i] = pos[indices[i]];
        }
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        seg::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch.at(nsample));

        std::vector<int> label_seq = speech::load_label_seq(label_batch.at(nsample), l_args.label_id);

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            auto f_var = comp_graph.var(la::tensor<double>(
                la::vector<double>(s.frames[i])));
            f_var->grad_needed = false;
            frame_ops.push_back(f_var);
        }

        std::shared_ptr<lstm::transcriber> trans
            = seg::make_transcriber(outer_layer, inner_layer, args, &l_args.gen);

        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { trans });
        frame_ops = (*trans)(lstm_var_tree, frame_ops);

        std::cout << "frames: " << s.frames.size() << " downsampled: " << frame_ops.size() << std::endl;

        if (frame_ops.size() < label_seq.size()) {
            continue;
        }

        // FIXME
        seg::make_graph(s, l_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);

        autodiff::eval(frame_mat, autodiff::eval_funcs);

        if (ebt::in(std::string("dropout"), args)) {
            // FIXME
            s.graph_data.weight_func = seg::make_weights(l_args.features, var_tree, frame_mat,
                dropout, &l_args.gen);
        } else {
            // FIXME
            s.graph_data.weight_func = seg::make_weights(l_args.features, var_tree, frame_mat);
        }

        seg::loss_func *loss_func;

        // FIXME
        loss_func = new seg::marginal_log_loss { s.graph_data, label_seq };

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / label_seq.size() << std::endl;

#if 0
        {
            seg::learning_args l_args2 = l_args;

            l_args2.param = tensor_tree::copy_tree(l_args.param);

            if (ebt::in(std::string("nn-param"), args)) {
                l_args2.nn_param = tensor_tree::copy_tree(l_args.nn_param);
            }

            auto vars = tensor_tree::leaves_pre_order(l_args2.nn_param);
            tensor_tree::get_tensor(vars[0]).data()[0] += 1e-8;

            seg::learning_sample s2 { l_args2 };

            s2.frames = s.frames;

            std::vector<int> label_seq2 = label_seq;

            autodiff::computation_graph comp_graph2;
            std::shared_ptr<tensor_tree::vertex> var_tree2
                = tensor_tree::make_var_tree(comp_graph2, l_args2.param);

            std::shared_ptr<tensor_tree::vertex> lstm_var_tree2;

            if (ebt::in(std::string("nn-param"), args)) {
                lstm_var_tree2 = tensor_tree::make_var_tree(comp_graph2, l_args2.nn_param);
            }

            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops2;
            for (int i = 0; i < s2.frames.size(); ++i) {
                auto f_var = comp_graph2.var(la::tensor<double>(
                    la::vector<double>(s2.frames[i])));
                f_var->grad_needed = false;
                frame_ops2.push_back(f_var);
            }

            if (ebt::in(std::string("nn-param"), args)) {
                std::shared_ptr<lstm::transcriber> trans2 = seg::make_transcriber(l_args2);

                if (ebt::in(std::string("logsoftmax"), args)) {
                    trans2 = std::make_shared<lstm::logsoftmax_transcriber>(
                        lstm::logsoftmax_transcriber { trans2 });
                    frame_ops2 = (*trans2)(lstm_var_tree2, frame_ops2);
                } else {
                    frame_ops2 = (*trans2)(lstm_var_tree2->children[0], frame_ops2);
                }
            }

            seg::make_graph(s2, l_args2, frame_ops2.size());

            auto frame_mat2 = autodiff::row_cat(frame_ops2);

            autodiff::eval(frame_mat2, autodiff::eval_funcs);

            if (ebt::in(std::string("dropout"), args)) {
                s2.graph_data.weight_func = seg::make_weights(l_args2.features, var_tree2, frame_mat2,
                    dropout, &l_args2.gen);
            } else {
                s2.graph_data.weight_func = seg::make_weights(l_args2.features, var_tree2, frame_mat2);
            }

            seg::marginal_log_loss loss_func2 { s2.graph_data, label_seq2 };

            double ell2 = loss_func2.loss();

            std::cout << vars.back()->name << " "
                << "numeric grad: " << (ell2 - ell) / 1e-8 << std::endl;
        }
#endif

        std::shared_ptr<tensor_tree::vertex> param_grad
            = seg::make_lstm_tensor_tree(outer_layer, inner_layer);

        if (ell > 0) {
            loss_func->grad();

            s.graph_data.weight_func->grad();

            autodiff::grad(frame_mat, autodiff::grad_funcs);
            tensor_tree::copy_grad(param_grad, lstm_var_tree);

            {
                auto vars = tensor_tree::leaves_pre_order(param_grad);
                std::cout << vars.back()->name << " "
                    << "analytic grad: " << tensor_tree::get_tensor(vars[0]).data()[0]
                    << std::endl;
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);

            double v1 = tensor_tree::get_tensor(vars[0]).data()[0];

            if (ebt::in(std::string("clip"), args)) {
                double n = tensor_tree::norm(param_grad);

                if (n > clip) {
                    tensor_tree::imul(param_grad, clip / n);

                    std::cout << "grad norm: " << n
                        << " clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            if (ebt::in(std::string("decay"), args)) {
                tensor_tree::rmsprop_update(param, param_grad,
                    opt_data, l_args.decay, l_args.step_size);
            } else if (ebt::in(std::string("momentum"), args)) {
                tensor_tree::const_step_update_momentum(param, param_grad,
                    opt_data, l_args.step_size, l_args.momentum);
            } else if (ebt::in(std::string("const-step-update"), args)) {
                tensor_tree::const_step_update(param, param_grad,
                    l_args.step_size);
            } else {
                tensor_tree::adagrad_update(param, param_grad,
                    opt_data, l_args.step_size);
            }

            double v2 = tensor_tree::get_tensor(vars[0]).data()[0];

            std::cout << "weight: " << v1 << " update: " << v2 - v1
                << " ratio: " << (v2 - v1) / v1 << std::endl;

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        double n = tensor_tree::norm(param);

        std::cout << "norm: " << n << std::endl;

        std::cout << std::endl;

        ++nsample;

        delete loss_func;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

    seg::save_lstm_param(outer_layer, inner_layer,
        param, output_param);
    seg::save_lstm_param(outer_layer, inner_layer,
        opt_data, output_opt_data);

}

