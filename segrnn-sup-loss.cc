#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"
#include "nn/lstm-frame.h"

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

    int min_seg;
    int max_seg;
    int stride;

    std::ifstream frame_batch;
    std::ifstream seg_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;
    std::vector<int> sils;

    int subsampling;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-sup-loss",
        "Learn segmental RNN",
        {
            {"frame-batch", "", true},
            {"seg-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"sil", "", true},
            {"logsoftmax", "", false},
            {"subsampling", "", false},
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

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(features, layer);
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

    frame_batch.open(args.at("frame-batch"));
    seg_batch.open(args.at("seg-batch"));

    subsampling = 0;
    if (ebt::in(std::string("subsampling"), args)) {
        subsampling = std::stoi(args.at("subsampling"));
    }

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
    for (auto& s: ebt::split(args.at("sil"), ",")) {
        sils.push_back(label_id.at(s));
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch);

        if (!frame_batch || !seg_batch) {
            break;
        }

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << segs.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < frames.size(); ++i) {
            auto f_var = comp_graph.var(la::tensor<double>(
                la::vector<double>(frames[i])));
            frame_ops.push_back(f_var);
        }

        std::shared_ptr<lstm::transcriber> trans;
        if (ebt::in(std::string("subsampling"), args)) {
            trans = lstm_frame::make_pyramid_transcriber(layer, 0.0, nullptr);
        } else {
            trans = lstm_frame::make_transcriber(layer, 0.0, nullptr);
        }

        if (ebt::in(std::string("logsoftmax"), args)) {
            trans = std::make_shared<lstm::logsoftmax_transcriber>(
                lstm::logsoftmax_transcriber { trans });
            frame_ops = (*trans)(var_tree->children[1], frame_ops);
        } else {
            frame_ops = (*trans)(var_tree->children[1]->children[0], frame_ops);
        }

        std::cout << "frames: " << frames.size() << " downsampled: " << frame_ops.size() << std::endl;

        if (frame_ops.size() < segs.size()) {
            ++nsample;
            continue;
        }

        seg::iseg_data graph_data;
        graph_data.fst = seg::make_graph(frame_ops.size(), label_id, id_label, min_seg, max_seg, stride);
        graph_data.topo_order = std::make_shared<std::vector<int>>(fst::topo_order(*graph_data.fst));

        auto frame_mat = autodiff::row_cat(frame_ops);

        graph_data.weight_func = seg::make_weights(features, var_tree->children[0], frame_mat);

        seg::loss_func *loss_func;

        std::vector<cost::segment<int>> gt_segs;
        for (auto& s: segs) {
            gt_segs.push_back(cost::segment<int> {(long)(s.start_time / std::pow(2, subsampling)),
                (long)(s.end_time / std::pow(2, subsampling)), label_id.at(s.label)});
        }

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

        ++nsample;

        delete loss_func;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

}

