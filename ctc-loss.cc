#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"
#include "seg/ctc.h"
#include "nn/lstm-frame.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "ctc-loss",
        "Compute CTC loss",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"type", "ctc,hmm1s,hmm2s", true}
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
    param = lstm_frame::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);
        std::vector<std::string> label_seq = speech::load_label_seq_batch(label_batch);

        if (!frame_batch || !label_batch) {
            break;
        }

        std::vector<int> label_id_seq;
        for (auto& s: label_seq) {
            label_id_seq.push_back(label_id.at(s));
        }

        std::cout << "sample: " << nsample + 1 << std::endl;
        std::cout << "gold len: " << label_seq.size() << std::endl;

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < frames.size(); ++i) {
            auto f_var = comp_graph.var(la::tensor<double>(
                la::vector<double>(frames[i])));
            f_var->grad_needed = false;
            frame_ops.push_back(f_var);
        }

        std::shared_ptr<lstm::transcriber> trans;

        if (ebt::in(std::string("subsampling"), args)) {
            trans = lstm_frame::make_pyramid_transcriber(layer, 0.0, nullptr);
        } else {
            trans = lstm_frame::make_transcriber(layer, 0.0, nullptr);
        }

        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { trans });
        frame_ops = (*trans)(lstm_var_tree, frame_ops);

        std::cout << "frames: " << frames.size() << " downsampled: " << frame_ops.size() << std::endl;

        if (frame_ops.size() < label_seq.size()) {
            continue;
        }

        ifst::fst graph_fst = ctc::make_frame_fst(frame_ops.size(), label_id, id_label);

        seg::iseg_data graph_data;
        graph_data.fst = std::make_shared<ifst::fst>(graph_fst);
        graph_data.weight_func = std::make_shared<ctc::label_weight>(ctc::label_weight(frame_ops));

        ifst::fst label_fst;

        if (args.at("type") == "ctc") {
            label_fst = ctc::make_label_fst(label_id_seq, label_id, id_label);
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

        std::cout << std::endl;

        ++nsample;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

    }

}

