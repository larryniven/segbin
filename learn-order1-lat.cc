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

    double step_size;
    double decay;
    double momentum;

    std::vector<std::string> features;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    double cost_scale;

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
            {"step-size", "", true},
            {"decay", "", false},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
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

    features = ebt::split(args.at("features"), ",");

    param = fscrf::make_tensor_tree(features);
    tensor_tree::load_tensor(param, args.at("param"));
    opt_data = fscrf::make_tensor_tree(features);
    tensor_tree::load_tensor(opt_data, args.at("opt-data"));

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
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 0;

    while (1) {

        ilat::fst lat = ilat::load_lattice(lat_batch, label_id);
        std::vector<segcost::segment<int>> gt_segs = util::load_segments(gt_batch, label_id);

        if (!lat_batch || !gt_batch) {
            break;
        }

        std::vector<std::vector<double>> frames;
        if (ebt::in(std::string("frame-batch"), args)) {
            frames = speech::load_frame_batch(frame_batch);
        }

        std::cout << lat.data->name << std::endl;

        fscrf::fscrf_data graph_data;
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(lat));
        graph_data.fst = std::make_shared<ilat::fst>(lat);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        if (ebt::in(std::string("frame-batch"), args)) {
            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
            for (int i = 0; i < frames.size(); ++i) {
                frame_ops.push_back(comp_graph.var(la::vector<double>(frames[i])));
            }
            auto frame_mat = autodiff::col_cat(frame_ops);
            autodiff::eval(frame_mat, autodiff::eval_funcs);
            graph_data.weight_func = fscrf::make_weights(features, var_tree, frame_mat);
        } else {
            graph_data.weight_func = fscrf::lat::make_weights(features, var_tree);
        }

        fscrf::loss_func *loss_func;

        if (args.at("loss") == "hinge-loss") {
            loss_func = new fscrf::hinge_loss { graph_data, gt_segs, sils, cost_scale };
        } else if (args.at("loss") == "log-loss") {
            loss_func = new fscrf::log_loss { graph_data, gt_segs, sils };
        }

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad = fscrf::make_tensor_tree(features);

        if (ell > 0) {
            loss_func->grad();

            graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            if (ebt::in(std::string("decay"), args)) {
                tensor_tree::rmsprop_update(param, param_grad, opt_data,
                    decay, step_size);
            } else if (ebt::in(std::string("momentum"), args)) {
                tensor_tree::const_step_update_momentum(param, param_grad, opt_data,
                    step_size, momentum);
            } else {
                tensor_tree::adagrad_update(param, param_grad, opt_data,
                    step_size);
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
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

    tensor_tree::save_tensor(param, output_param);
    tensor_tree::save_tensor(opt_data, output_opt_data);

}

