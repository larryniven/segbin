#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct forced_alignment_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    fscrf::inference_args l_args;

    std::unordered_map<std::string, std::string> args;

    forced_alignment_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "forced-align-order1-full",
        "Align with segmental CRF",
        {
            {"frame-batch", "", false},
            {"label-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", false},
            {"features", "", true},
            {"label", "", true},
            {"frames", "", false},
            {"segs", "", false}
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

    forced_alignment_env env { args };

    env.run();

    return 0;
}

forced_alignment_env::forced_alignment_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    label_batch.open(args.at("label-batch"));

    fscrf::parse_inference_args(l_args, args);
}

void forced_alignment_env::run()
{
    ebt::Timer timer;

    int i = 0;

    while (1) {

        fscrf::sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);
        std::vector<std::string> label_seq = util::load_label_seq(label_batch);

        if (!label_batch) {
            break;
        }

        fscrf::make_graph(s, l_args);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, l_args.nn_param);
            pred_var_tree = make_var_tree(comp_graph, l_args.pred_param);
        }

        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops;

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans = fscrf::make_transcriber(l_args);
            feat_ops = (*trans)(lstm_var_tree, frame_ops);
            pred_nn = rnn::make_pred_nn(pred_var_tree, feat_ops);
            feat_ops = pred_nn.logprob;
        } else {
            feat_ops = frame_ops;
        }

        auto frame_mat = autodiff::row_cat(feat_ops);

        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = fscrf::make_weights(l_args.features, var_tree, frame_mat);

        std::vector<int> label_seq_id;
        for (auto& s: label_seq) {
            label_seq_id.push_back(l_args.label_id.at(s));
        }

        ilat::fst label_fst = fscrf::make_label_fst(label_seq_id, l_args.label_id, l_args.id_label);

        ilat::fst& graph_fst = *s.graph_data.fst;

        ilat::lazy_pair_mode1 composed_fst { label_fst, graph_fst };

        fscrf::fscrf_pair_data pair_data;
        pair_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);
        pair_data.weight_func = std::make_shared<fscrf::mode2_weight>(
            fscrf::mode2_weight { s.graph_data.weight_func });
        pair_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));

        fscrf::fscrf_pair_fst pair { pair_data };

        fst::forward_one_best<fscrf::fscrf_pair_fst> one_best;
        for (auto& i: composed_fst.initials()) {
            one_best.extra[i] = fst::forward_one_best<fscrf::fscrf_pair_fst>::extra_data
                { std::make_tuple(-1, -1), 0 };
        }
        one_best.merge(pair, *pair_data.topo_order);

        std::vector<std::tuple<int, int>> edges = one_best.best_path(pair);

        fscrf::fscrf_fst graph { s.graph_data };

        if (ebt::in(std::string("frames"), args)) {
            std::cout << i + 1 << ".phn" << std::endl;
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
            std::cout << i + 1 << ".phn" << std::endl;
            for (auto& e: edges) {
                int tail_time = graph.time(graph.tail(std::get<1>(e)));
                int head_time = graph.time(graph.head(std::get<1>(e)));

                std::cout << tail_time << " " << head_time << " " << l_args.id_label.at(pair.output(e)) << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            for (auto& e: edges) {
                std::cout << l_args.id_label.at(pair.output(e)) << " "
                    << "(" << graph.time(graph.head(std::get<1>(e))) << ") ";
            }
            std::cout << std::endl;
        }

        ++i;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

    }

}

