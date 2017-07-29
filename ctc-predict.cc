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
            {"dyer-lstm", "", false},
            {"rmdup", "", false},
            {"beam-search", "", false},
            {"beam-width", "", false},
            {"type", "ctc,hmm1s,hmm2s", true},
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
    if (ebt::in(std::string("dyer-lstm"), args)) {
        param = lstm_frame::make_dyer_tensor_tree(layer);
    } else {
        param = lstm_frame::make_tensor_tree(layer);
    }
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

        std::shared_ptr<lstm::transcriber> trans;

        if (ebt::in(std::string("subsampling"), args)) {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_transcriber(param, 0.0, nullptr, true);
            } else {
                trans = lstm_frame::make_transcriber(param, 0.0, nullptr, true);
            }
        } else {
            if (ebt::in(std::string("dyer-lstm"), args)) {
                trans = lstm_frame::make_dyer_transcriber(param, 0.0, nullptr, false);
            } else {
                trans = lstm_frame::make_transcriber(param, 0.0, nullptr, false);
            }
        }

        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { (int) label_id.size(), trans });

        lstm::trans_seq_t input_seq;
        input_seq.nframes = frames.size();
        input_seq.batch_size = 1;
        input_seq.dim = frames.front().size();
        input_seq.feat = input;
        input_seq.mask = nullptr;

        lstm::trans_seq_t output_seq = (*trans)(lstm_var_tree, input_seq);

        std::shared_ptr<autodiff::op_t> logprob = output_seq.feat;

        auto& logprob_t = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        ifst::fst graph_fst = ctc::make_frame_fst(logprob_t.size(0), label_id, id_label);

        auto& logprob_mat = logprob_t.as_matrix();
        auto logprob_m = autodiff::weak_var(logprob, 0, std::vector<unsigned int> { logprob_mat.rows(), logprob_mat.cols() });

        seg::iseg_data graph_data;
        graph_data.fst = std::make_shared<ifst::fst>(graph_fst);
        graph_data.weight_func = std::make_shared<ctc::label_weight>(
            ctc::label_weight(logprob_m));

        seg::seg_fst<seg::iseg_data> graph { graph_data };

        if (ebt::in(std::string("beam-search"), args)) {
            int beam_width = std::stoi(args.at("beam-width"));

            ctc::beam_search<seg::seg_fst<seg::iseg_data>> beam_search;

            beam_search.search(graph, label_id.at("<blk>"), beam_width);

            std::unordered_map<int, double> path_score;

            if (args.at("type") == "ctc" && beam_search.heap.size() > 0) {
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
        } else if (args.at("type") == "ctc" && ebt::in(std::string("rmdup"), args)) {
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
        } else if (args.at("type") == "hmm1s" && ebt::in(std::string("rmdup"), args)) {
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

                if (last != o_i) {
                    std::cout << id_label.at(o_i) << " ";
                    last = o_i;
                }
            }
            std::cout << "(" << nsample << ".dot)" << std::endl;
        } else if (args.at("type") == "hmm2s" && ebt::in(std::string("rmdup"), args)) {
            fst::forward_one_best<seg::seg_fst<seg::iseg_data>> one_best;

            for (auto& i: graph.initials()) {
                one_best.extra[i] = {-1, 0};
            }

            auto topo_order = fst::topo_order(graph);

            one_best.merge(graph, topo_order);

            std::vector<int> path = one_best.best_path(graph);

            for (int i = 0; i < path.size(); ++i) {
                auto& o_i = graph.output(path[i]);

                if (!ebt::endswith(id_label.at(o_i), "-")) {
                    std::cout << id_label.at(o_i) << " ";
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

