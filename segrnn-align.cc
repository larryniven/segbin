#include "seg/seg-util.h"
#include "util/speech.h"
#include "util/util.h"
#include "fst/fst-algo.h"
#include "seg/seg.h"
#include <fstream>
#include "nn/lstm-frame.h"

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

struct alignment_env {

    std::vector<std::string> features;

    std::ifstream frame_batch;
    std::ifstream label_batch;

    int max_seg;
    int min_seg;
    int stride;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    int seed;
    std::default_random_engine gen;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    alignment_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-align",
        "Align with segmental RNN",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"frames", "", false},
            {"segs", "", false},
            {"subsampling", "", false},
            {"logsoftmax", "", false},
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

    alignment_env env { args };

    env.run();

    return 0;
}

alignment_env::alignment_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    features = ebt::split(args.at("features"), ",");

    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

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

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    gen = std::default_random_engine{seed};

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void alignment_env::run()
{
    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);
        std::vector<int> label_seq = speech::load_label_seq_batch(label_batch, label_id);

        if (!frame_batch || !label_batch) {
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
                trans = lstm_frame::make_dyer_transcriber(param->children[1]->children[0], 0.0, nullptr, true);
            } else {
                trans = lstm_frame::make_transcriber(param->children[1]->children[0], 0.0, nullptr, true);
            }
        } else {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_transcriber(param->children[1]->children[0], 0.0, nullptr, false);
            } else {
                trans = lstm_frame::make_transcriber(param->children[1]->children[0], 0.0, nullptr, false);
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

        ifst::fst label_fst = seg::make_label_fst(label_seq, label_id, id_label);

        ifst::fst& graph_fst = *graph_data.fst;

        fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst> composed_fst { label_fst, graph_fst };

        seg::pair_iseg_data pair_data;
        pair_data.fst = std::make_shared<fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst>>(composed_fst);
        pair_data.weight_func = std::make_shared<seg::mode2_weight>(
            seg::mode2_weight { graph_data.weight_func });
        pair_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));

        seg::seg_fst<seg::pair_iseg_data> pair { pair_data };

        fst::forward_one_best<seg::seg_fst<seg::pair_iseg_data>> one_best;
        for (auto& i: composed_fst.initials()) {
            one_best.extra[i] = fst::forward_one_best<seg::seg_fst<seg::pair_iseg_data>>::extra_data
                { std::make_tuple(-1, -1), 0 };
        }
        one_best.merge(pair, *pair_data.topo_order);

        std::vector<std::tuple<int, int>> edges = one_best.best_path(pair);

        seg::seg_fst<seg::iseg_data> graph { graph_data };

        if (ebt::in(std::string("frames"), args)) {
            std::cout << nsample + 1 << ".phn" << std::endl;
            int t = 0;
            for (auto& e: edges) {
                int head_time = graph.time(graph.head(std::get<1>(e)));
                for (int j = t; j < head_time; ++j) {
                    std::cout << id_label.at(pair.output(e)) << std::endl;
                }
                t = head_time;
            }
            std::cout << "." << std::endl;
        } else if (ebt::in(std::string("segs"), args)) {
            std::cout << nsample + 1 << ".phn" << std::endl;
            for (auto& e: edges) {
                int tail_time = graph.time(graph.tail(std::get<1>(e)));
                int head_time = graph.time(graph.head(std::get<1>(e)));

                if (ebt::in(std::string("subsampling"), args)) {
                    tail_time *= 2 * (layer - 1);
                    head_time *= 2 * (layer - 1);
                }

                std::cout << tail_time << " " << head_time
                    << " " << id_label.at(pair.output(e)) << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            for (auto& e: edges) {
                std::cout << id_label.at(pair.output(e)) << " "
                    << "(" << graph.time(graph.head(std::get<1>(e))) << ") ";
            }
            std::cout << std::endl;
        }

        ++nsample;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

}

