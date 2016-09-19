#include "seg/iscrf_e2e.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/align.h"
#include "seg/util.h"
#include "seg/pair_scrf.h"
#include "nn/lstm.h"
#include "nn/tensor_tree.h"
#include <random>
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    std::ifstream lattice_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    iscrf::e2e::learning_args l_args;

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
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"l2", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"dropout-seed", "", false},
            {"frame-softmax", "", false},
            {"clip", "", false},
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

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
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

    iscrf::e2e::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 1;

    std::default_random_engine gen { l_args.dropout_seed };

    while (1) {

        iscrf::learning_sample s { l_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> label_seq = util::load_label_seq(label_batch);

        if (!label_batch || !frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = make_var_tree(comp_graph, l_args.nn_param);
        std::shared_ptr<tensor_tree::vertex> pred_var_tree
            = make_var_tree(comp_graph, l_args.pred_param);
        lstm::stacked_bi_lstm_nn_t nn;
        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops
            = iscrf::e2e::make_feat(comp_graph, lstm_var_tree, pred_var_tree,
                nn, pred_nn, frames, gen, l_args);

        auto order = autodiff::topo_order(feat_ops);
        autodiff::eval(order, autodiff::eval_funcs);

        std::vector<std::vector<double>> feats;
        for (auto& o: feat_ops) {
            auto& f = autodiff::get_output<la::vector<double>>(o);
            feats.push_back(std::vector<double> {f.data(), f.data() + f.size()});
        }

        s.frames = feats;

        if (ebt::in(std::string("lattice-batch"), args)) {
            ilat::fst lat = ilat::load_lattice(lattice_batch, l_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            iscrf::make_lattice(lat, s, l_args);
        } else {
            iscrf::make_graph(s, l_args);
        }

        // compute loss

        double num = 0;
        double denom = 0;

        ilat::fst label_seq_fst = iscrf::make_label_seq_fst(label_seq,
            l_args.label_id, l_args.id_label);

        ilat::lazy_pair_mode2 label_graph_fst { *s.graph_data.fst, label_seq_fst };

        iscrf::second_order::pair_scrf_data<scrf::dense_vec> label_graph_data;

        label_graph_data.fst = std::make_shared<ilat::lazy_pair_mode2>(label_graph_fst);
        label_graph_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(label_graph_fst));

        scrf::feat_dim_alloc label_alloc { l_args.labels };

        scrf::composite_feature<ilat::pair_fst, scrf::dense_vec> label_feat_func
            = iscrf::second_order::make_feat<scrf::dense_vec, iscrf::second_order::dense::pair_fst_lexicalizer>(
                label_alloc, l_args.features, s.frames, l_args.args);

        label_graph_data.feature_func = std::make_shared<
            scrf::composite_feature<ilat::pair_fst, scrf::dense_vec>>(label_feat_func);

        scrf::composite_weight<ilat::pair_fst> label_weight_func;

        label_weight_func.weights.push_back(
            std::make_shared<scrf::cached_linear_score<ilat::pair_fst, scrf::dense_vec>>(
                scrf::cached_linear_score<ilat::pair_fst, scrf::dense_vec> {
                    l_args.param, label_graph_data.feature_func }));

        label_graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::pair_fst>>(label_weight_func);

        scrf::scrf_fst<iscrf::second_order::pair_scrf_data<scrf::dense_vec>> label_graph { label_graph_data };

        fst::forward_log_sum<decltype(label_graph)> label_forward;

        label_forward.merge(label_graph, *label_graph_data.topo_order);

        auto rev_label_topo_order = *label_graph_data.topo_order;
        std::reverse(rev_label_topo_order.begin(), rev_label_topo_order.end());

        fst::backward_log_sum<decltype(label_graph)> label_backward;
        label_backward.merge(label_graph, rev_label_topo_order);

        for (auto& i: label_graph.initials()) {
            std::cout << "num forward: " << label_backward.extra.at(i) << std::endl;
        }

        for (auto& f: label_graph.finals()) {
            std::cout << "num backward: " << label_forward.extra.at(f) << std::endl;
        }

        num = label_forward.extra.at(label_graph.finals().front());

        using comp_feat = scrf::composite_feature<ilat::fst, scrf::dense_vec>;

        comp_feat graph_feat_func
            = iscrf::make_feat(s.graph_alloc, l_args.features, s.frames, l_args.args);

        scrf::composite_weight<ilat::fst> graph_weight;
        graph_weight.weights.push_back(std::make_shared<scrf::cached_linear_score<ilat::fst, scrf::dense_vec>>(
            scrf::cached_linear_score<ilat::fst, scrf::dense_vec>(l_args.param,
            std::make_shared<comp_feat>(graph_feat_func))));

        s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(graph_weight);
        s.graph_data.feature_func = std::make_shared<comp_feat>(graph_feat_func);

        scrf::scrf_fst<iscrf::iscrf_data> graph { s.graph_data };

        fst::forward_log_sum<decltype(graph)> graph_forward;

        graph_forward.merge(graph, *s.graph_data.topo_order);

        auto rev_topo_order = *s.graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        fst::backward_log_sum<decltype(graph)> graph_backward;
        graph_backward.merge(graph, rev_topo_order);

        for (auto& i: graph.initials()) {
            std::cout << "denom forward: " << graph_backward.extra.at(i) << std::endl;
        }

        for (auto& f: graph.finals()) {
            std::cout << "denom backward: " << graph_forward.extra.at(f) << std::endl;
        }

        denom = graph_forward.extra.at(graph.finals().front());

        double ell = -num + denom;

        std::cout << "loss: " << ell << std::endl;

#if 0
        scrf::dense_vec param_grad;
        std::shared_ptr<tensor_tree::vertex> nn_param_grad
            = lstm::make_stacked_bi_lstm_tensor_tree(l_args.layer);
        std::shared_ptr<tensor_tree::vertex> pred_grad
            = nn::make_pred_tensor_tree();

        if (ell > 0) {
            // param grad

            std::vector<std::vector<double>> frame_grad;
            frame_grad.resize(feats.size());
            for (int i = 0; i < feats.size(); ++i) {
                frame_grad[i].resize(feats[i].size());
            }

            std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>> feat_func
                = iscrf::e2e::filter_feat_with_frame_grad(s.graph_data);

            // frame grad

            for (int i = 0; i < frame_grad.size(); ++i) {
                feat_ops[i]->grad = std::make_shared<la::vector<double>>(
                    la::vector<double>(frame_grad[i]));
            }

            autodiff::grad(order, autodiff::grad_funcs);

            tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);

            if (ebt::in(std::string("frame-softmax"), args)) {
                tensor_tree::copy_grad(pred_grad, pred_var_tree);
            }

            if (ebt::in(std::string("clip"), args)) {
                double n = tensor_tree::norm(nn_param_grad);
                if (n > l_args.clip) {
                    tensor_tree::imul(nn_param_grad, l_args.clip / n);
                    std::cout << "grad norm: " << n << " clip: " << l_args.clip << " gradient clipped" << std::endl;
                }
            }

            if (ebt::in(std::string("l2"), l_args.args)) {
                scrf::dense_vec p = l_args.param;
                scrf::imul(p, l_args.l2);
                scrf::iadd(param_grad, p);

                std::shared_ptr<tensor_tree::vertex> nn_p = tensor_tree::copy_tree(l_args.nn_param);
                tensor_tree::imul(nn_p, l_args.l2);
                tensor_tree::iadd(nn_param_grad, nn_p);
            }

            double v1 = get_matrix(l_args.nn_param->children[0]->children[0]->children[0])(0, 0);
            double w1 = l_args.param.class_vec[2](0);

            if (ebt::in(std::string("decay"), args)) {
                scrf::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.decay, l_args.step_size);

                if (!ebt::in(std::string("freeze-encoder"), l_args.args)) {
                    tensor_tree::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                        l_args.decay, l_args.step_size);

                    if (ebt::in(std::string("frame-softmax"), args)) {
                        tensor_tree::rmsprop_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                            l_args.decay, l_args.step_size);
                    }
                }
            } else {
                scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size);

                if (!ebt::in(std::string("freeze-encoder"), l_args.args)) {
                    tensor_tree::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                        l_args.step_size);

                    if (ebt::in(std::string("frame-softmax"), args)) {
                        tensor_tree::adagrad_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                            l_args.step_size);
                    }
                }
            }

            double v2 = get_matrix(l_args.nn_param->children[0]->children[0]->children[0])(0, 0);
            double w2 = l_args.param.class_vec[2](0);

            std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;
            std::cout << "weight: " << w1 << " update: " << w2 - w1 << " rate: " << (w2 - w1) / w1 << std::endl;

            if (i % save_every == 0) {
                scrf::save_vec(l_args.param, "param-last");
                scrf::save_vec(l_args.opt_data, "opt-data-last");
                iscrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, "nn-param-last");
                iscrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
            }

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }
#endif

        std::cout << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_vec(l_args.param, output_param);
    scrf::save_vec(l_args.opt_data, output_opt_data);
    iscrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, output_nn_param);
    iscrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, output_nn_opt_data);

}
