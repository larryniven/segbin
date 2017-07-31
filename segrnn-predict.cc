#include "seg/seg-util.h"
#include "speech/speech.h"
#include "fst/fst-algo.h"
#include "nn/lstm-frame.h"
#include <fstream>

std::shared_ptr<tensor_tree::vertex> make_dyer_tensor_tree(
    std::vector<std::string> const& features,
    int layer)
{
    tensor_tree::vertex root;

    root.children.push_back(seg::make_tensor_tree(features));
    root.children.push_back(lstm_frame::make_dyer_tensor_tree(layer));

    return std::make_shared<tensor_tree::vertex>(root);
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
    std::vector<std::string> const& features,
    int layer)
{
    tensor_tree::vertex root;

    root.children.push_back(seg::make_tensor_tree(features));
    root.children.push_back(lstm_frame::make_tensor_tree(layer));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct prediction_env {

    std::vector<std::string> features;

    std::ifstream frame_batch;

    int min_seg;
    int max_seg;
    int stride;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-predict",
        "Predict with segmental RNN",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"features", "", true},
            {"subsampling", "", false},
            {"logsoftmax", "", false},
            {"dyer-lstm", "", false},
            {"label", "", true},
            {"print-path", "", false},
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

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    features = ebt::split(args.at("features"), ",");

    frame_batch.open(args.at("frame-batch"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    if (ebt::in(std::string("dyer-lstm"), args)) {
        param = make_dyer_tensor_tree(features, layer);
    } else {
        param = make_tensor_tree(features, layer);
    }
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

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

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void prediction_env::run()
{
    int nsample = 1;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
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

        std::shared_ptr<autodiff::op_t> input
            = comp_graph.var(la::cpu::weak_tensor<double>(
                frame_cat.data(), { nframes, ndim }));

        input->grad_needed = false;

        std::shared_ptr<lstm::transcriber> trans;

        if (ebt::in(std::string("subsampling"), args)) {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_transcriber(param->children[1], 0.0, nullptr, true);
            } else {
                trans = lstm_frame::make_transcriber(param->children[1], 0.0, nullptr, true);
            }
        } else {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_transcriber(param->children[1], 0.0, nullptr, false);
            } else {
                trans = lstm_frame::make_transcriber(param->children[1], 0.0, nullptr, false);
            }
        }

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        lstm::trans_seq_t output_seq = (*trans)(var_tree->children[1]->children[0], input_seq);

        if (ebt::in(std::string("logsoftmax"), args)) {
            lstm::fc_transcriber fc_trans { (int) label_id.size() };
            lstm::logsoftmax_transcriber logsoftmax_trans;
            auto score = fc_trans(var_tree->children[1]->children[1], output_seq);

            output_seq = logsoftmax_trans(nullptr, score);
        }

        std::shared_ptr<autodiff::op_t> hidden = output_seq.feat;

        auto& hidden_t = autodiff::get_output<la::cpu::tensor_like<double>>(hidden);

        auto& hidden_mat = hidden_t.as_matrix();
        auto hidden_m = autodiff::weak_var(hidden, 0,
            std::vector<unsigned int> { hidden_mat.rows(), hidden_mat.cols() });

        seg::iseg_data graph_data;
        graph_data.fst = seg::make_graph(hidden_t.size(0), label_id, id_label, min_seg, max_seg, stride);
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(*graph_data.fst));

        graph_data.weight_func = seg::make_weights(features, var_tree->children[0], hidden_m);

        seg::seg_fst<seg::iseg_data> graph { graph_data };

        std::vector<int> path = fst::shortest_path(graph, *graph_data.topo_order);

        if (ebt::in(std::string("print-path"), args)) {
            std::cout << nsample << ".txt" << std::endl;
            for (auto& e: path) {
                std::cout << graph.time(graph.tail(e))
                    << " " << graph.time(graph.head(e))
                    << " " << id_label.at(graph.output(e)) << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            for (auto& e: path) {
                std::cout << id_label.at(graph.output(e)) << " ";
            }
            std::cout << "(" << nsample << ".dot)";
            std::cout << std::endl;
        }

        ++nsample;
    }
}

