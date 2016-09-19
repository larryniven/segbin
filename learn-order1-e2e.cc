#include "seg/iscrf_e2e.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "autodiff/autodiff.h"
#include "nn/lstm.h"
#include <fstream>
#include <random>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gold_batch;

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
            {"gold-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"l2", "", false},
            {"step-size", "", true},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"decay", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"loss", "", true},
            {"cost-scale", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"dropout-seed", "", false},
            {"clip", "", false},
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

    gold_batch.open(args.at("gold-batch"));

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
    int i = 1;

    std::default_random_engine gen { l_args.dropout_seed };

    while (1) {

        iscrf::learning_sample s { l_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        s.gold_segs = util::load_segments(gold_batch, l_args.label_id);

        if (!gold_batch) {
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

        iscrf::make_min_cost_gold(s, l_args);

        iscrf::parameterize(s, l_args);

        double gold_cost = 0;

        iscrf::iscrf_fst gold { s.gold_data };
    
        std::cout << "gold path: ";
        for (auto& e: gold.edges()) {
            std::cout << l_args.id_label[gold.output(e)] << " ";
            gold_cost += cost(s.gold_data, e);
        }
        std::cout << std::endl;
    
        std::cout << "gold cost: " << gold_cost << std::endl;

        std::shared_ptr<scrf::loss_func_with_frame_grad<scrf::dense_vec, ilat::fst>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s.graph_data.cost_func);
            s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            loss_func = std::make_shared<hinge_loss>(
                hinge_loss { s.gold_data, s.graph_data });

            hinge_loss const& loss
                = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            iscrf::iscrf_fst gold { s.gold_data };

            std::cout << "gold: ";
            for (auto& e: gold.edges()) {
                std::cout << l_args.id_label[gold.output(e)] << " ";
                gold_weight += gold.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            iscrf::iscrf_fst graph_path { loss.graph_path };

            std::cout << "cost aug: ";
            for (auto& e: graph_path.edges()) {
                std::cout << l_args.id_label[graph_path.output(e)] << " ";
                graph_weight += graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl;
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << s.gold_data.fst->edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::dense_vec param_grad;
        std::shared_ptr<tensor_tree::vertex> nn_param_grad
            = lstm::make_stacked_bi_lstm_tensor_tree(l_args.layer);
        std::shared_ptr<tensor_tree::vertex> pred_grad
            = nn::make_pred_tensor_tree();

        if (ell > 0) {
            param_grad = loss_func->param_grad();

            std::vector<std::vector<double>> frame_grad;
            frame_grad.resize(feats.size());
            for (int i = 0; i < feats.size(); ++i) {
                frame_grad[i].resize(feats[i].size());
            }

            std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>> feat_func
                = iscrf::e2e::filter_feat_with_frame_grad(s.graph_data);

            loss_func->frame_grad(*feat_func, frame_grad, l_args.param);

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

                if (ebt::in(std::string("frame-softmax"), args)) {
                    std::shared_ptr<tensor_tree::vertex> pred_p = tensor_tree::copy_tree(l_args.pred_param);
                    tensor_tree::imul(pred_p, l_args.l2);
                    tensor_tree::iadd(pred_grad, pred_p);
                }
            }

            double v1 = get_matrix(l_args.nn_param->children[0]->children[0]->children[0])(0, 0);

            if (ebt::in(std::string("decay"), args)) {
                scrf::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.decay, l_args.step_size);
                tensor_tree::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.decay, l_args.step_size);

                if (ebt::in(std::string("frame-softmax"), args)) {
                    tensor_tree::rmsprop_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                        l_args.decay, l_args.step_size);
                }
            } else {
                scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size);
                tensor_tree::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.step_size);

                if (ebt::in(std::string("frame-softmax"), args)) {
                    tensor_tree::adagrad_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                        l_args.step_size);
                }
            }

            double v2 = get_matrix(l_args.nn_param->children[0]->children[0]->children[0])(0, 0);

            std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (i % save_every == 0) {
            scrf::save_vec(l_args.param, "param-last");
            scrf::save_vec(l_args.opt_data, "opt-data-last");
            iscrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, "nn-param-last");
            iscrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
        }

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

