#include "seg/seg-util.h"
#include "speech/speech.h"
#include "fst/fst-algo.h"
#include "seg/seg.h"
#include <fstream>

struct alignment_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    int inner_layer;
    int outer_layer;
    std::shared_ptr<tensor_tree::vertex> nn_param;

    seg::inference_args l_args;

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
            {"nn-param", "", false},
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
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    label_batch.open(args.at("label-batch"));

    if (ebt::in(std::string("nn-param"), args)) {
        std::tie(outer_layer, inner_layer, nn_param)
            = seg::load_lstm_param(args.at("nn-param"));
    }

    seg::parse_inference_args(l_args, args);
}

void alignment_env::run()
{
    int nsample = 0;

    while (1) {

        seg::sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);
        std::vector<std::string> label_seq = speech::load_label_seq(label_batch);

        if (!label_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = tensor_tree::make_var_tree(comp_graph, nn_param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::tensor<double>(la::vector<double>(s.frames[i]))));
        }

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans
                = seg::make_transcriber(outer_layer, inner_layer, args, nullptr);

            if (ebt::in(std::string("logsoftmax"), args)) {
                trans = std::make_shared<lstm::logsoftmax_transcriber>(
                    lstm::logsoftmax_transcriber { trans });
                frame_ops = (*trans)(lstm_var_tree, frame_ops);
            } else {
                frame_ops = (*trans)(lstm_var_tree->children[0], frame_ops);
            }
        }

        seg::make_graph(s, l_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);

        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = seg::make_weights(l_args.features, var_tree, frame_mat);

        std::vector<int> label_seq_id;
        for (auto& s: label_seq) {
            label_seq_id.push_back(l_args.label_id.at(s));
        }

        ifst::fst label_fst = seg::make_label_fst(label_seq_id, l_args.label_id, l_args.id_label);

        ifst::fst& graph_fst = *s.graph_data.fst;

        fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst> composed_fst { label_fst, graph_fst };

        seg::pair_iseg_data pair_data;
        pair_data.fst = std::make_shared<fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst>>(composed_fst);
        pair_data.weight_func = std::make_shared<seg::mode2_weight>(
            seg::mode2_weight { s.graph_data.weight_func });
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

        seg::seg_fst<seg::iseg_data> graph { s.graph_data };

        if (ebt::in(std::string("frames"), args)) {
            std::cout << nsample + 1 << ".phn" << std::endl;
            int t = 0;
            for (auto& e: edges) {
                int head_time = graph.time(graph.head(std::get<1>(e)));
                for (int j = t; j < head_time; ++j) {
                    std::cout << l_args.id_label.at(pair.output(e)) << std::endl;
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
                    tail_time *= 4;
                    head_time *= 4;
                }

                std::cout << tail_time << " " << head_time
                    << " " << l_args.id_label.at(pair.output(e)) << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            for (auto& e: edges) {
                std::cout << l_args.id_label.at(pair.output(e)) << " "
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

