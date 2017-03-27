#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"
#include "nn/lstm-frame.h"
#include "nn/nn.h"

std::vector<std::string> to_frame_labels(int nframes,
    std::vector<speech::segment> const& segs);

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
    std::vector<std::string> const& features,
    int layer)
{
    tensor_tree::vertex root;

    root.children.push_back(seg::make_tensor_tree(features));
    root.children.push_back(lstm_frame::make_tensor_tree(layer));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct learning_env {

    std::vector<std::string> features;

    speech::batch_indices frame_batch;
    speech::batch_indices seg_batch;

    int max_seg;
    int min_seg;
    int stride;

    std::string output_param;
    std::string output_opt_data;

    int seed;
    std::default_random_engine gen;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    double dropout;
    double clip;
    double step_size;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::shared_ptr<tensor_tree::optimizer> opt;

    double lambda;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-frame-mll-learn",
        "Learn segmental RNN",
        {
            {"frame-batch", "", false},
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
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,const-step-momentum,adagrad,rmsprop,adam", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
            {"momentum", "", false},
            {"beta1", "", false},
            {"beta2", "", false},
            {"lambda", "", false}
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

    frame_batch.open(args.at("frame-batch"));
    seg_batch.open(args.at("seg-batch"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(features, layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

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

    id_label = speech::load_label_set(args.at("label"));
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
    std::getline(opt_data_ifs, line);
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    lambda = 0.5;
    if (ebt::in(std::string("lambda"), args)) {
        lambda = std::stod(args.at("lambda"));
        assert(0 <= lambda && lambda <= 1);
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));

        std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch.at(nsample));

        std::vector<int> label_seq;
        for (auto& s: segs) {
            label_seq.push_back(label_id.at(s.label));
        }

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < frames.size(); ++i) {
            auto f_var = comp_graph.var(la::tensor<double>(
                la::vector<double>(frames[i])));
            f_var->grad_needed = false;
            frame_ops.push_back(f_var);
        }

        std::shared_ptr<lstm::transcriber> trans
            = lstm_frame::make_pyramid_transcriber(layer, dropout, &gen);

        frame_ops = (*trans)(var_tree->children[1]->children[0], frame_ops);

        std::cout << "frames: " << frames.size() << " downsampled: " << frame_ops.size() << std::endl;

        if (frame_ops.size() < label_seq.size()) {
            continue;
        }

        seg::iseg_data graph_data;
        graph_data.fst = seg::make_graph(frame_ops.size(), label_id, id_label, min_seg, max_seg, stride);
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(*graph_data.fst));

        auto frame_mat = autodiff::row_cat(frame_ops);

        if (ebt::in(std::string("dropout"), args)) {
            graph_data.weight_func = seg::make_weights(features, var_tree->children[0], frame_mat,
                dropout, &gen);
        } else {
            graph_data.weight_func = seg::make_weights(features, var_tree->children[0], frame_mat);
        }

        seg::loss_func *loss_func;

        ifst::fst label_fst = seg::make_label_fst(label_seq, label_id, id_label);

        loss_func = new seg::marginal_log_loss { graph_data, label_fst };

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / label_seq.size() << std::endl;

        // frame classifier

        std::vector<std::string> frame_labels = to_frame_labels(frames.size(), segs);

        assert(frame_labels.size() == frames.size());

        std::shared_ptr<lstm::transcriber> logprob_trans
            = std::make_shared<lstm::logsoftmax_transcriber>(
                lstm::logsoftmax_transcriber { nullptr });
        std::vector<std::shared_ptr<autodiff::op_t>> logprob
            = (*logprob_trans)(var_tree->children[1], frame_ops);

        double frame_loss_sum = 0;
        double nframes = 0;

        int freq = std::round(double(frame_labels.size()) / logprob.size());

        for (int t = 0; t < logprob.size(); ++t) {
            auto& pred = autodiff::get_output<la::tensor<double>>(logprob[t]);
            la::tensor<double> gold;
            gold.resize({(unsigned int)(label_id.size() - 1)});

            if (t * freq >= frame_labels.size()) {
                break;
            }

            if (frame_labels[t * freq] != "unk") {
                gold({label_id.at(frame_labels[t * freq]) - 1}) = 1;
            }

            nn::log_loss frame_loss { gold, pred };
            logprob[t]->grad = std::make_shared<la::tensor<double>>(frame_loss.grad(1 - lambda));

            if (std::isnan(frame_loss.loss())) {
                std::cerr << "loss is nan" << std::endl;
                exit(1);
            } else {
                frame_loss_sum += frame_loss.loss();
                nframes += 1;
            }
        }

        std::cout << "frame loss: " << frame_loss_sum / nframes << std::endl;

        auto logprob_order = autodiff::topo_order(logprob, frame_ops);
        autodiff::grad(logprob_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> param_grad
            = make_tensor_tree(features, layer);

        if (ell > 0) {
            loss_func->grad(lambda);

            graph_data.weight_func->grad();

            autodiff::grad(frame_mat, autodiff::grad_funcs);
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

                if (n > clip) {
                    tensor_tree::imul(param_grad, clip / n);

                    std::cout << "grad norm: " << n
                        << " clip: " << clip << " gradient clipped" << std::endl;
                }
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

    std::ofstream param_ofs { args.at("output-param") };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { args.at("output-opt-data") };
    opt_data_ofs << layer << std::endl;
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

std::vector<std::string> to_frame_labels(int nframes,
    std::vector<speech::segment> const& segs)
{
    std::vector<std::string> result;
    int seg_index = 0;

    for (int i = 0; i < nframes; ++i) {
        while (seg_index < segs.size() && i >= segs.at(seg_index).end_time) {
            ++seg_index;
        }

        if (seg_index == segs.size() || i < segs.at(seg_index).start_time) {
            result.push_back("unk");
        } else if (segs.at(seg_index).start_time <= i && i < segs.at(seg_index).end_time) {
            result.push_back(segs.at(seg_index).label);
        }
    }

    return result;
}

