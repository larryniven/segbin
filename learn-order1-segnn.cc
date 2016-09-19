#include "seg/iscrf_segnn.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "autodiff/autodiff.h"
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

    iscrf::segnn::learning_args l_args;

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
            {"stride", "", false},
            {"param", "", true},
            {"opt-data", "", true},
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

    iscrf::segnn::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    int i = 1;

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

        s.frames = frames;

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

        autodiff::computation_graph comp_graph;
        l_args.nn = segnn::make_nn(comp_graph, l_args.nn_param);

        iscrf::make_min_cost_gold(s, l_args);

        iscrf::segnn::parameterize(s, l_args);

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
                std::cout << l_args.id_label[gold.output(e)] << " " << gold.weight(e) << " ";
                gold_weight += gold.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            iscrf::iscrf_fst graph_path { loss.graph_path };

            std::cout << "cost aug: ";
            for (auto& e: graph_path.edges()) {
                std::cout << l_args.id_label[graph_path.output(e)] << " " << graph_path.weight(e) << " ";
                graph_weight += graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl;

        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

#if 0
        {
            iscrf::segnn::learning_args l_args2 = l_args;
            l_args2.nn_param.label_embedding(l_args2.label_id.at("sil"), 0) += 1e-8;
            iscrf::learning_sample s2 { l_args2 };
            s2.frames = frames;
            s2.gold_segs = s.gold_segs;
            iscrf::make_graph(s2, l_args2);
            autodiff::computation_graph comp_graph2;
            l_args2.nn = segnn::make_nn(comp_graph2, l_args2.nn_param);
            iscrf::make_min_cost_gold(s2, l_args2);
            iscrf::segnn::parameterize(s2, l_args2);

            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s2.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s2.graph_data.cost_func);
            s2.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(
                weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;
            hinge_loss loss2 { s2.gold_data, s2.graph_data };

            std::cout << "numeric grad: " << (loss2.loss() - loss_func->loss()) / 1e-8 << std::endl;
        }
#endif

        std::cout << "gold segs: " << s.gold_data.fst->edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::dense_vec param_grad;
        segnn::param_t nn_param_grad;

        segnn::resize_as(nn_param_grad, l_args.nn_param);

        if (ell > 0) {
            param_grad = loss_func->param_grad();

            // compute gradient

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            hinge_loss const& loss
                = *dynamic_cast<hinge_loss*>(loss_func.get());

            iscrf::iscrf_fst gold { s.gold_data };

            scrf::dense_vec neg_param = l_args.param;
            scrf::imul(neg_param, -1);

            std::shared_ptr<segnn::segnn_feat> segnn_feat = std::dynamic_pointer_cast<segnn::segnn_feat>(
                s.gold_data.feature_func->features[0]);

            for (auto& e: gold.edges()) {
                segnn_feat->grad(neg_param, *s.gold_data.fst, e);

                segnn::iadd(nn_param_grad, segnn_feat->gradient);
            }

            segnn_feat = std::dynamic_pointer_cast<segnn::segnn_feat>(
                s.graph_data.feature_func->features[0]);

            iscrf::iscrf_fst graph_path { loss.graph_path };

            for (auto& e: graph_path.edges()) {
                segnn_feat->grad(l_args.param, *s.graph_data.fst, e);

                segnn::iadd(nn_param_grad, segnn_feat->gradient);
            }

            // std::cout << "sil id: " << l_args.label_id.at("sil") << std::endl;
            // std::cout << "analytic grad: " << nn_param_grad.label_embedding(l_args.label_id.at("sil"), 0) << std::endl;

            scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                l_args.step_size);
            segnn::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.step_size);

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (i % save_every == 0) {
            scrf::save_vec(l_args.param, "param-last");
            scrf::save_vec(l_args.opt_data, "opt-data-last");
            segnn::save_nn_param(l_args.nn_param, "nn-param-last");
            segnn::save_nn_param(l_args.nn_opt_data, "nn-opt-data-last");
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
    segnn::save_nn_param(l_args.nn_param, output_nn_param);
    segnn::save_nn_param(l_args.nn_opt_data, output_nn_opt_data);

}

