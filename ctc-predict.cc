#include "seg/seg-util.h"
#include "speech/speech.h"
#include <fstream>
#include "ebt/ebt.h"
#include "seg/loss.h"
#include "seg/ctc.h"
#include "nn/lstm-frame.h"

struct prediction_env {

    std::ifstream frame_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "ctc-predict",
        "Find 1-best path",
        {
            {"frame-batch", "", false},
            {"param", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"rmdup", "", false},
            {"beam-search", "", false},
            {"beam-width", "", false},
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
    frame_batch.open(args.at("frame-batch"));

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

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

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

        ifst::fst graph_fst = ctc::make_frame_fst(frame_ops.size(), label_id, id_label);

        seg::iseg_data graph_data;
        graph_data.fst = std::make_shared<ifst::fst>(graph_fst);
        graph_data.weight_func = std::make_shared<ctc::label_weight>(
            ctc::label_weight(frame_ops));

        seg::seg_fst<seg::iseg_data> graph { graph_data };

        if (ebt::in(std::string("beam-search"), args)) {
            int beam_width = std::stoi(args.at("beam-width"));

            ctc::beam_search<seg::seg_fst<seg::iseg_data>> beam_search;

            beam_search.search(graph, label_id.at("<blk>"), beam_width);

            std::unordered_map<int, double> path_score;

            if (beam_search.heap.size() > 0) {
                double inf = std::numeric_limits<double>::infinity();
                double max = -inf;
                int argmax = -1;

                for (auto& k: beam_search.path_score) {
                    if (!ebt::in(k.first.second, path_score)) {
                        path_score[k.first.second] = -inf;
                    }

                    path_score[k.first.second] = ebt::log_add(path_score[k.first.second], k.second);
                }

                for (auto& k: path_score) {
                    if (k.second > max) {
                        max = k.second;
                        argmax = k.first;
                    }
                }

                for (auto& p: beam_search.id_seq[argmax]) {
                    std::cout << id_label.at(p) << " ";
                }
                std::cout << "(" << nsample << ".dot)" << std::endl;
            } else {
                std::cout << "(" << nsample << ".dot)" << std::endl;
            }
        } else if (ebt::in(std::string("rmdup"), args)) {
            fst::forward_one_best<seg::seg_fst<seg::iseg_data>> one_best;

            for (auto& i: graph.initials()) {
                one_best.extra[i] = {-1, 0};
            }

            auto topo_order = fst::topo_order(graph);

            one_best.merge(graph, topo_order);

            std::vector<int> path = one_best.best_path(graph);

            int last = -1;
            for (int i = 0; i < path.size(); ++i) {
                int o_i = graph.output(path[i]);

                if (last != o_i && o_i != label_id.at("<blk>")) {
                    std::cout << id_label.at(o_i) << " ";
                    last = o_i;
                } else if (i >= 1 && graph.output(path[i-1]) == label_id.at("<blk>")
                        && o_i != label_id.at("<blk>")) {
                    std::cout << id_label.at(o_i) << " ";
                    last = o_i;
                }
            }
            std::cout << "(" << nsample << ".dot)" << std::endl;

        } else {
            fst::forward_one_best<seg::seg_fst<seg::iseg_data>> one_best;

            for (auto& i: graph.initials()) {
                one_best.extra[i] = {-1, 0};
            }

            auto topo_order = fst::topo_order(graph);

            one_best.merge(graph, topo_order);

            std::vector<int> path = one_best.best_path(graph);

            for (auto& e: path) {
                std::cout << id_label.at(graph.output(e)) << " ";
            }
            std::cout << "(" << nsample << ".dot)" << std::endl;
        }

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

        ++nsample;

    }

}

