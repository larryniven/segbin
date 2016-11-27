#include "ebt/ebt.h"
#include <fstream>
#include "seg/ctc.h"
#include "seg/util.h"
#include "nn/lstm-tensor-tree.h"
#include "speech/speech.h"
#include "autodiff/autodiff.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, std::string> args;

    ctc::learning_args l_args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-ctc",
        "Learn LSTM with CTC",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"momentum", "", false},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"dropout-seed", "", false},
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
    ctc::parse_learning_args(l_args, args);

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
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 1;

    std::default_random_engine gen { l_args.dropout_seed };

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> label_seq = util::load_label_seq(label_batch);

        if (!label_batch || !frame_batch) {
            break;
        }

        std::cout << "sample: " << i << std::endl;
        std::cout << "segs: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = make_var_tree(comp_graph, l_args.nn_param);
        std::shared_ptr<tensor_tree::vertex> pred_var_tree
            = make_var_tree(comp_graph, l_args.pred_param);
        lstm::stacked_bi_lstm_nn_t nn;
        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops
            = ctc::make_feat(comp_graph, lstm_var_tree, pred_var_tree,
                nn, pred_nn, frames, gen, l_args);

        auto order = autodiff::topo_order(feat_ops);
        autodiff::eval(order, autodiff::eval_funcs);

        std::vector<std::vector<double>> feats;
        for (auto& o: feat_ops) {
            auto& f = autodiff::get_output<la::vector<double>>(o);
            feats.push_back(std::vector<double> {f.data(), f.data() + f.size()});
        }

        ilat::fst label_fst = ctc::make_label_fst(label_seq, l_args.label_id, l_args.id_label);
        ilat::fst frame_fst = ctc::make_frame_fst(feats, l_args.label_id, l_args.id_label);

        ilat::lazy_pair_mode1 composed_fst { label_fst, frame_fst };

        std::vector<std::tuple<int, int>> fst_order = fst::topo_order(composed_fst);
        fst::forward_log_sum<ilat::lazy_pair_mode1> forward;
        forward.merge(composed_fst, fst_order);

        auto rev_fst_order = fst_order;
        std::reverse(rev_fst_order.begin(), rev_fst_order.end());
        fst::backward_log_sum<ilat::lazy_pair_mode1> backward;
        backward.merge(composed_fst, rev_fst_order);

        for (auto& i: composed_fst.initials()) {
            std::cout << "forward: " << backward.extra.at(i) << std::endl;
        }

        for (auto& f: composed_fst.finals()) {
            std::cout << "backward: " << forward.extra.at(f) << std::endl;
        }

        double Z = forward.extra.at(composed_fst.finals().front());

        std::cout << "loss: " << -Z << " " << " normalized loss: " << -Z / frames.size() << std::endl;

        std::vector<std::vector<double>> feat_grad;
        feat_grad.resize(feats.size());
        for (int i = 0; i < feats.size(); ++i) {
            feat_grad[i].resize(feats[i].size());
        }

        std::vector<std::vector<std::tuple<int, int>>> edge_index;
        edge_index.resize(frame_fst.edges().size());
        auto edges = composed_fst.edges();

        std::cout << "edges: " << edges.size() << std::endl;

        for (auto& e: edges) {
            if (composed_fst.output(e) == 0) {
                continue;
            }

            edge_index[std::get<1>(e)].push_back(e);
        }

        for (int e2 = 0; e2 < edge_index.size(); ++e2) {
            int u = frame_fst.tail(e2);
            int v = frame_fst.head(e2);

            if (u == v) {
                continue;
            }

            double sum = 0;

            for (auto& e: edge_index[e2]) {
                auto tail = composed_fst.tail(e);
                auto head = composed_fst.head(e);

                if (!ebt::in(tail, forward.extra) || !ebt::in(head, backward.extra)) {
                    continue;
                }

                double g = forward.extra.at(tail) + composed_fst.weight(e)
                    + backward.extra.at(head) - Z;

                sum += -std::exp(g);
            }

            feat_grad[u][frame_fst.output(e2) - 1] = sum;
        }

        for (int i = 0; i < feat_grad.size(); ++i) {
            feat_ops[i]->grad = std::make_shared<la::vector<double>>(
                la::vector<double>(feat_grad[i]));
        }

        autodiff::grad(order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> nn_param_grad
            = lstm::make_stacked_bi_lstm_tensor_tree(l_args.layer);
        std::shared_ptr<tensor_tree::vertex> pred_grad
            = nn::make_pred_tensor_tree();

        tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);
        tensor_tree::copy_grad(pred_grad, pred_var_tree);

        if (ebt::in(std::string("clip"), args)) {
            double n1 = tensor_tree::norm(nn_param_grad);
            double n2 = tensor_tree::norm(pred_grad);
            double n = std::sqrt(n1 * n1 + n2 * n2);
            if (n > l_args.clip) {
                tensor_tree::imul(nn_param_grad, l_args.clip / n);
                tensor_tree::imul(pred_grad, l_args.clip / n);
                std::cout << "grad norm: " << n << " clip: " << l_args.clip << " gradient clipped" << std::endl;
            }
        }

        double v1 = get_matrix(l_args.pred_param->children[0])(0, 0);

        if (ebt::in(std::string("momentum"), args)) {
            tensor_tree::const_step_update_momentum(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.momentum, l_args.step_size);

            tensor_tree::const_step_update_momentum(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                l_args.momentum, l_args.step_size);
        } else if (ebt::in(std::string("decay"), args)) {
            tensor_tree::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.decay, l_args.step_size);

            tensor_tree::rmsprop_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                l_args.decay, l_args.step_size);
        } else {
            tensor_tree::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.step_size);

            tensor_tree::adagrad_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                l_args.step_size);
        }

        double v2 = get_matrix(l_args.pred_param->children[0])(0, 0);

        std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;

        if (i % save_every == 0) {
            ctc::save_lstm_param(l_args.nn_param, l_args.pred_param, "nn-param-last");
            ctc::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    ctc::save_lstm_param(l_args.nn_param, l_args.pred_param, output_param);
    ctc::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, output_opt_data);

}
