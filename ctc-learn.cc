#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"
#include "seg/ctc.h"
#include "nn/lstm-frame.h"
#include <sstream>

using namespace std::string_literals;

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices label_batch;

    std::string output_param;
    std::string output_opt_data;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    int cell_dim;

    double step_size;
    double dropout;
    double clip;

    int seed;
    std::default_random_engine gen;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;

    std::vector<int> indices;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "ctc-learn",
        "Train RNN with CTC",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"seed", "", false},
            {"subsampling", "", false},
            {"shuffle", "", false},
            {"dyer-lstm", "", false},
            {"opt", "const-step,rmsprop,adagrad,adam", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
            {"momentum", "", false},
            {"beta1", "", false},
            {"beta2", "", false},
            {"type", "ctc,ctc-1b,hmm1s,hmm2s", true},
            {"random-state", "", false}
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
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    if (ebt::in(std::string("dyer-lstm"), args)) {
        param = lstm_frame::make_dyer_tensor_tree(layer);
    } else {
        param = lstm_frame::make_tensor_tree(layer);
    }
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    if (ebt::in(std::string("dyer-lstm"), args)) {
        cell_dim = tensor_tree::get_tensor(param->children[0]
            ->children[0]->children[0]->children[0]).size(1) / 3;
    } else {
        cell_dim = tensor_tree::get_tensor(param->children[0]
            ->children[0]->children[0]->children[0]).size(1) / 4;
    }

    step_size = std::stod(args.at("step-size"));

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stod(args.at("seed"));
    }

    gen = std::default_random_engine { seed };

    if (ebt::in("random-state"s, args)) {
        std::istringstream iss { args.at("random-state") };
        iss >> gen;
    }

    if (ebt::in(std::string("shuffle"), args)) {
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
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt{param, step_size});
    } else if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt{param, step_size, decay});
    } else if (args.at("opt") == "adam") {
        double beta1 = std::stod(args.at("beta1"));
        double beta2 = std::stod(args.at("beta2"));
        opt = std::make_shared<tensor_tree::adam_opt>(
            tensor_tree::adam_opt{param, step_size, beta1, beta2});
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<std::string> label_seq = speech::load_label_seq_batch(label_batch.at(nsample));

        std::vector<int> label_id_seq;
        for (auto& s: label_seq) {
            label_id_seq.push_back(label_id.at(s));
        }

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<double> frame_cat;
        frame_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            frame_cat.insert(frame_cat.end(), frames[i].begin(), frames[i].end());
        }

        unsigned int nframes = frames.size();
        unsigned int ndim = frames.front().size();

        std::shared_ptr<autodiff::op_t> input
            = comp_graph.var(la::cpu::weak_tensor<double>(
                frame_cat.data(), { nframes, ndim }));

        input->grad_needed = false;

        std::cout << "random: " << gen << std::endl;

        std::shared_ptr<lstm::transcriber> trans;

        if (ebt::in(std::string("subsampling"), args)) {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_pyramid_transcriber(layer, dropout, &gen);
            } else {
                trans = lstm_frame::make_pyramid_transcriber(layer, dropout, &gen);
            }
        } else {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_transcriber(layer, dropout, &gen);
            } else {
                trans = lstm_frame::make_transcriber(layer, dropout, &gen);
            }
        }

        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { trans });

        std::shared_ptr<autodiff::op_t> logprob;
        std::shared_ptr<autodiff::op_t> ignore;
        std::tie(logprob, ignore) = (*trans)(frames.size(), 1, cell_dim, lstm_var_tree, input);

        auto& logprob_t = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        std::cout << "frames: " << frames.size() << " downsampled: " << logprob_t.size(0) << std::endl;

        if (logprob_t.size(0) < label_seq.size()) {
            continue;
        }

        ifst::fst graph_fst = ctc::make_frame_fst(logprob_t.size(0), label_id, id_label);

        auto& logprob_mat = logprob_t.as_matrix();
        auto logprob_m = autodiff::weak_var(logprob, 0, std::vector<unsigned int> { logprob_mat.rows(), logprob_mat.cols() });

        seg::iseg_data graph_data;
        graph_data.fst = std::make_shared<ifst::fst>(graph_fst);
        graph_data.weight_func = std::make_shared<ctc::label_weight>(ctc::label_weight(logprob_m));

        ifst::fst label_fst;

        if (args.at("type") == "ctc") {
            label_fst = ctc::make_label_fst(label_id_seq, label_id, id_label);
        } else if (args.at("type") == "ctc-1b") {
            label_fst = ctc::make_label_fst_1b(label_id_seq, label_id, id_label);
        } else if (args.at("type") == "hmm1s") {
            label_fst = ctc::make_label_fst_hmm1s(label_id_seq, label_id, id_label);
        } else if (args.at("type") == "hmm2s") {
            label_fst = ctc::make_label_fst_hmm2s(label_id_seq, label_id, id_label);
        } else {
            std::cout << "unknown type " << args.at("type") << std::endl;
            exit(1);
        }

        ctc::loss_func loss {graph_data, label_fst};

        double ell = loss.loss();

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / label_seq.size() << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad;

        if (ebt::in(std::string("dyer-lstm"), args)) {
            param_grad = lstm_frame::make_dyer_tensor_tree(layer);
        } else {
            param_grad = lstm_frame::make_tensor_tree(layer);
        }

        if (ell > 0) {
            loss.grad();
            graph_data.weight_func->grad();

            auto topo_order = autodiff::natural_topo_order(comp_graph);
            autodiff::guarded_grad(topo_order, autodiff::grad_funcs);
            tensor_tree::copy_grad(param_grad, lstm_var_tree);

            {
                auto vars = tensor_tree::leaves_pre_order(param_grad);
                std::cout << vars.back()->name << " "
                    << "analytic grad: " << tensor_tree::get_tensor(vars[0]).data()[0]
                    << std::endl;
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars = tensor_tree::leaves_pre_order(param);

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

#if 0
            std::cout << "sample: " << indices[nsample] << std::endl;

            std::ofstream param_ofs { "param-debug" };
            param_ofs << layer << std::endl;
            tensor_tree::save_tensor(param, param_ofs);
            param_ofs.close();

            std::ofstream opt_data_ofs { "opt-data-debug" };
            opt_data_ofs << layer << std::endl;
            opt->save_opt_data(opt_data_ofs);
            opt_data_ofs.close();

            exit(1);
#endif
        }

        double n = tensor_tree::norm(param);

        std::cout << "norm: " << n << std::endl;

        std::cout << std::endl;

        ++nsample;

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
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

