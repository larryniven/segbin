#include "seg/seg-util.h"
#include "util/speech.h"
#include "util/util.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"

struct learning_env {

    std::vector<std::string> features;

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    int max_seg;
    int min_seg;
    int stride;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    int seed;
    std::default_random_engine gen;

    double dropout;
    double step_size;
    double clip;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "seglin-learn",
        "Learn a linear segmental model",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"features", "", true},
            {"label", "", true},
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"nsample", "", false},
            {"opt", "const-step,rmsprop,adagrad", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
            {"momentum", "", false},
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
    features = ebt::split(args.at("features"), ",");

    param = seg::make_tensor_tree(features);
    tensor_tree::load_tensor(param, args.at("param"));

    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    step_size = std::stod(args.at("step-size"));

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    stride = 1;
    if (ebt::in(std::string("stride"), args)) {
        stride = std::stoi(args.at("stride"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    gen = std::default_random_engine{seed};

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::vector<int> indices;
        indices.resize(frame_batch.pos.size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            frame_batch.pos[i] = pos[indices[i]];
        }

        pos = label_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            label_batch.pos[i] = pos[indices[i]];
        }
    }

    if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt{param, step_size});
    } else if (args.at("opt") == "const-step-momentum") {
        double momentum = std::stod(args.at("momentum"));
        opt = std::make_shared<tensor_tree::const_step_momentum_opt>(
            tensor_tree::const_step_momentum_opt{param, step_size, momentum});
    } else if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt{param, step_size, decay});
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt{param, step_size});
    } else if (args.at("opt") == "adam") {
        double beta1 = std::stod(args.at("beta1"));
        double beta2 = std::stod(args.at("beta2"));
        opt = std::make_shared<tensor_tree::adam_opt>(
            tensor_tree::adam_opt{param, step_size, beta1, beta2});
    } else {
        std::cout << "unknown optimizer " << args.at("opt") << std::endl;
        exit(1);
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));

        std::vector<int> label_seq = speech::load_label_seq_batch(label_batch.at(nsample), label_id);

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        std::cout << "frames: " << frames.size() << std::endl;

        if (frames.size() < label_seq.size()) {
            ++nsample;
            continue;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<double> frame_cat;
        frame_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            frame_cat.insert(frame_cat.end(), frames[i].begin(), frames[i].end());
        }

        unsigned int nframes = frames.size();
        unsigned int ndim = frames.front().size();

        std::shared_ptr<autodiff::op_t> frame_mat = comp_graph.var(la::cpu::weak_tensor<double>(
            frame_cat.data(), { nframes, ndim }));

        if (ebt::in(std::string("dropout"), args)) {
            auto d_mask = autodiff::dropout_mask(frame_mat, dropout, gen);
            frame_mat = autodiff::emul(d_mask, frame_mat);
        }

        seg::iseg_data graph_data;
        graph_data.fst = seg::make_graph(frames.size(), label_id, id_label, min_seg, max_seg, stride);
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(*graph_data.fst));

        if (ebt::in(std::string("dropout"), args)) {
            graph_data.weight_func = seg::make_weights(features, var_tree, frame_mat,
                dropout, &gen);
        } else {
            graph_data.weight_func = seg::make_weights(features, var_tree, frame_mat);
        }

        seg::loss_func *loss_func;

        ifst::fst label_fst = seg::make_label_fst(label_seq, label_id, id_label);

        loss_func = new seg::marginal_log_loss { graph_data, label_fst };

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / label_seq.size() << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad = seg::make_tensor_tree(features);

        if (ell > 0) {
            loss_func->grad();

            graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            {
                auto vars = tensor_tree::leaves_pre_order(param_grad);
                std::cout << vars.back()->name << " "
                    << "analytic grad: " << tensor_tree::get_tensor(vars[0]).data()[0]
                    << std::endl;
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars
                = tensor_tree::leaves_pre_order(param);

            double v1 = tensor_tree::get_tensor(vars[0]).data()[0];

            if (ebt::in(std::string("clip"), args)) {
                double n = tensor_tree::norm(param_grad);

                std::cout << "grad norm: " << n;

                if (n > clip) {
                    tensor_tree::axpy(param_grad, clip / n - 1, param_grad);

                    std::cout << " clip: " << clip << " gradient clipped";
                }

                std::cout << std::endl;
            }

            opt->update(param_grad);

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

        if (ebt::in(std::string("nsample"), args) && nsample == std::stoi(args.at("nsample"))) {
            break;
        }

    }

    tensor_tree::save_tensor(param, output_param);

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

