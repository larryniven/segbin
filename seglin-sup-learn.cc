#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"

struct learning_env {

    std::vector<std::string> features;

    speech::batch_indices frame_batch;
    speech::batch_indices seg_batch;

    int max_seg;
    int min_seg;
    int stride;

    int subsampling;

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
    std::vector<int> sils;

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
            {"seg-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"features", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"sil", "", true},
            {"dropout", "", false},
            {"seed", "", false},
            {"subsampling", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,rmsprop,adagrad", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
            {"momentum", "", false},
            {"beta1", "", false},
            {"beta2", "", false},
            {"loss", "hinge-loss,log-loss", true},
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
    seg_batch.open(args.at("seg-batch"));

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

    subsampling = 0;
    if (ebt::in(std::string("subsampling"), args)) {
        subsampling = std::stoi(args.at("subsampling"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    gen = std::default_random_engine{seed};

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
    for (auto& s: ebt::split(args.at("sil"), ",")) {
        sils.push_back(label_id.at(s));
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

        pos = seg_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            seg_batch.pos[i] = pos[indices[i]];
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
        double beta1 = std::stod("beta1");
        double beta2 = std::stod("beta2");
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

        std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch.at(nsample));

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << segs.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        if (ebt::in(std::string("dropout"), args)) {
            for (int i = 0; i < frames.size(); ++i) {
                auto f_var = comp_graph.var(la::tensor<double>(
                    la::vector<double>(frames[i])));
                auto mask = autodiff::dropout_mask(f_var, dropout, gen);
                frame_ops.push_back(autodiff::emul(f_var, mask));
            }
        } else {
            for (int i = 0; i < frames.size(); ++i) {
                auto f_var = comp_graph.var(la::tensor<double>(
                    la::vector<double>(frames[i])));
                frame_ops.push_back(f_var);
            }
        }

        std::cout << "frames: " << frame_ops.size() << std::endl;

        if (frame_ops.size() < segs.size()) {
            ++nsample;
            continue;
        }

        seg::iseg_data graph_data;
        graph_data.fst = seg::make_graph(frame_ops.size(), label_id, id_label, min_seg, max_seg, stride);
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(*graph_data.fst));

        auto frame_mat = autodiff::row_cat(frame_ops);

        if (ebt::in(std::string("dropout"), args)) {
            graph_data.weight_func = seg::make_weights(features, var_tree, frame_mat,
                dropout, &gen);
        } else {
            graph_data.weight_func = seg::make_weights(features, var_tree, frame_mat);
        }

        std::vector<cost::segment<int>> gt_segs;
        for (auto& s: segs) {
            gt_segs.push_back(cost::segment<int> {(long)(s.start_time / std::pow(2, subsampling)),
                (long)(s.end_time / std::pow(2, subsampling)), label_id.at(s.label)});
        }

        seg::loss_func *loss_func;

        if (args.at("loss") == "log-loss") {
            loss_func = new seg::log_loss { graph_data, gt_segs, sils };
        } else if (args.at("loss") == "hinge-loss") {
            loss_func = new seg::hinge_loss { graph_data, gt_segs, sils };
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / segs.size() << std::endl;

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
                    tensor_tree::imul(param_grad, clip / n);

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

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

    tensor_tree::save_tensor(param, output_param);

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

