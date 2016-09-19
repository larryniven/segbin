#include "seg/iscrf_e2e.h"
#include "seg/iscrf_e2e_lstm2d.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/align.h"
#include "seg/util.h"
#include "nn/residual.h"
#include <random>
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    std::ifstream lattice_batch;

    scrf::dense_vec align_param;
    lstm::db_lstm2d_param_t align_nn_param;

    int save_every;
    int update_align_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    iscrf::e2e_lstm2d::learning_args l_args;

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
            {"stride", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"align-param", "", true},
            {"align-nn-param", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"update-align-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"loss", "", true},
            {"cost-scale", "", false},
            {"label", "", true},
            {"even-init", "", false}
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

    update_align_every = std::numeric_limits<int>::max();
    if (ebt::in(std::string("update-align-every"), args)) {
        update_align_every = std::stoi(args.at("update-align-every"));
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

    iscrf::e2e_lstm2d::parse_learning_args(l_args, args);

    align_param = scrf::load_dense_vec(args.at("align-param"));
    align_nn_param = lstm::load_db_lstm2d_param(args.at("align-nn-param"));
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 1;

    while (1) {

        iscrf::learning_sample s { l_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> label_seq = util::load_label_seq(label_batch);

        if (!label_batch || !frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;
        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(comp_graph.var(la::vector<double>(frames[i])));
        }

        lstm::db_lstm2d_nn_t nn = lstm::make_db_lstm2d_nn(comp_graph, l_args.nn_param, inputs);

        auto topo_order = autodiff::topo_order(nn.layer.back().output);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        std::vector<std::vector<double>> outputs;
        for (int i = 0; i < nn.layer.back().output.size(); ++i) {
            auto& v = autodiff::get_output<la::vector<double>>(nn.layer.back().output[i]);
            outputs.push_back(std::vector<double> { v.data(), v.data() + v.size() });
        }

        s.frames = outputs;

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

        if (ebt::in(std::string("even-init"), args) && i <= update_align_every) {
            iscrf::make_even_gold(label_seq, s, l_args);
        } else {
            iscrf::make_alignment_gold(align_param, label_seq, s, l_args);
        }

        iscrf::parameterize(s, l_args);

        iscrf::iscrf_fst gold { s.gold_data };

        std::cout << "gold path: ";
        for (auto& e: gold.edges()) {
            std::cout << l_args.id_label[gold.output(e)] << " ("
                << gold.time(gold.head(e)) << ") ";
        }
        std::cout << std::endl;
    
        std::shared_ptr<scrf::loss_func_with_frame_grad<scrf::dense_vec, ilat::fst>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s.graph_data.cost_func);
            s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            loss_func = std::make_shared<hinge_loss>(hinge_loss { s.gold_data, s.graph_data });

            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: gold.edges()) {
                std::cout << l_args.id_label[gold.output(e)] << " " << gold.weight(e) << " (" << gold.time(gold.head(e)) << ") ";
                gold_weight += gold.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            iscrf::iscrf_fst graph_path { loss.graph_path };

            std::cout << "cost aug: ";
            for (auto& e: graph_path.edges()) {
                std::cout << l_args.id_label[graph_path.output(e)] << " " << graph_path.weight(e) << " (" << graph_path.time(graph_path.head(e)) << ") ";
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

        if (ell > 0) {
            scrf::dense_vec param_grad = loss_func->param_grad();

            std::vector<std::vector<double>> frame_grad;
            frame_grad.resize(s.frames.size());
            for (int i = 0; i < s.frames.size(); ++i) {
                frame_grad[i].resize(s.frames[i].size());
            }

            std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>> feat_func
                = iscrf::e2e::filter_feat_with_frame_grad(s.graph_data);

            loss_func->frame_grad(*feat_func, frame_grad, l_args.param);

            for (int i = 0; i < frame_grad.size(); ++i) {
                nn.layer.back().output[i]->grad = std::make_shared<la::vector<double>>(frame_grad[i]);
            }

            autodiff::grad(topo_order, autodiff::grad_funcs);

            lstm::db_lstm2d_param_t nn_param_grad = copy_db_lstm2d_grad(nn);

            if (ebt::in(std::string("decay"), l_args.args)) {
                scrf::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.decay, l_args.step_size);
                lstm::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.decay, l_args.step_size);
            } else {
                scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size);
                lstm::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                    l_args.step_size);
            }

            if (i % save_every == 0) {
                scrf::save_vec(l_args.param, "param-last");
                scrf::save_vec(l_args.opt_data, "opt-data-last");
                lstm::save_db_lstm2d_param(l_args.nn_param, "nn-param-last");
                lstm::save_db_lstm2d_param(l_args.nn_opt_data, "nn-opt-data-last");
            }

            if (i % update_align_every == 0) {
                align_param = l_args.param;
                align_nn_param = l_args.nn_param;

                std::cout << std::endl;
                std::cout << "update align param" << std::endl;
            }

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

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
    lstm::save_db_lstm2d_param(l_args.nn_param, output_nn_param);
    lstm::save_db_lstm2d_param(l_args.nn_opt_data, output_nn_opt_data);

}

