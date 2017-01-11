#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream lat_batch;
    std::ifstream label_batch;

    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    double step_size;
    double decay;
    double momentum;

    std::vector<std::string> features;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::shared_ptr<ilat::fst> lm;

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
        "lat-order2-learn",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"lat-batch", "", true},
            {"label-batch", "", true},
            {"lm", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"cost-scale", "", false},
            {"label", "", true},
            {"const-step-update", "", false}
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
    label_batch.open(args.at("label-batch"));

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

    std::ifstream lm_stream { args.at("lm") };
    lm = std::make_shared<ilat::fst>(ilat::load_arpa_lm(lm_stream, label_id));
    lm_stream.close();

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
        std::vector<int> label_seq = util::load_label_seq(label_batch, label_id);

        if (!lat_batch) {
            break;
        }

        std::vector<std::vector<double>> frames;
        if (ebt::in(std::string("frame-batch"), args)) {
            frames = speech::load_frame_batch(frame_batch);
        }

        std::cout << lat.data->name << std::endl;

        ilat::add_eps_loops(lat);

        ilat::lazy_pair_mode1 composed_fst { lat, *lm };

        fscrf::fscrf_pair_data graph_data;
        graph_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));
        graph_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        graph_data.weight_func = fscrf::make_pair_weights(features, var_tree, frames);

        fscrf::loss_func *loss_func;

        loss_func = new fscrf::marginal_log_loss_pair { graph_data, label_seq };

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad = fscrf::make_tensor_tree(features);

        auto vars = tensor_tree::leaves_pre_order(param);
        auto& t = tensor_tree::get_tensor(vars[1]);

        double w1 = t({0});

        if (ell > 0) {
            loss_func->grad();

            graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            auto grad_vars = tensor_tree::leaves_pre_order(param_grad);
            auto& g = tensor_tree::get_tensor(grad_vars[1]);
            std::cout << "analytic grad: " << g({0}) << std::endl;

            if (ebt::in(std::string("decay"), args)) {
                tensor_tree::rmsprop_update(param, param_grad, opt_data,
                    decay, step_size);
            } else if (ebt::in(std::string("momentum"), args)) {
                tensor_tree::const_step_update_momentum(param, param_grad, opt_data,
                    step_size, momentum);
            } else if (ebt::in(std::string("const-step-update"), args)) {
                tensor_tree::const_step_update(param, param_grad, step_size);
            } else {
                tensor_tree::adagrad_update(param, param_grad, opt_data, step_size);
            }

        }

        double w2 = t({0});

        std::cout << "weight: " << w1 << " update: " << w2 - w1 / w1 << std::endl;

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

