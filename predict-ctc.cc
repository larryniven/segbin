#include "ebt/ebt.h"
#include <fstream>
#include "seg/ctc.h"
#include "seg/util.h"
#include "speech/speech.h"
#include "autodiff/autodiff.h"

struct prediction_env {

    std::ifstream frame_batch;

    std::unordered_map<std::string, std::string> args;

    ctc::inference_args i_args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-ctc",
        "Predict with LSTM posteriors",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"log-prob", "", false},
            {"frame-pred", "", false}
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
    ctc::parse_inference_args(i_args, args);

    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int i = 1;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> lstm_var_tree
            = make_var_tree(comp_graph, i_args.nn_param);
        std::shared_ptr<tensor_tree::vertex> pred_var_tree
            = make_var_tree(comp_graph, i_args.pred_param);
        lstm::stacked_bi_lstm_nn_t nn;
        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (auto& f: frames) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(f)));
        }

        nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, lstm::bi_lstm_builder{});

        pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

        auto order = autodiff::topo_order(pred_nn.logprob);
        autodiff::eval(order, autodiff::eval_funcs);

        std::vector<std::vector<double>> feats;
        for (auto& o: pred_nn.logprob) {
            auto& f = autodiff::get_output<la::vector<double>>(o);
            feats.push_back(std::vector<double> {f.data(), f.data() + f.size()});
        }

        ilat::fst frame_fst = ctc::make_frame_fst(feats, i_args.label_id, i_args.id_label);

        if (ebt::in(std::string("log-prob"), args)) {
            std::cout << i << ".logp" << std::endl;
            for (int i = 0; i < feats.size(); ++i) {
                for (int d = 0; d < feats[i].size(); ++d) {
                    if (d == 0) {
                        std::cout << feats[i][d];
                    } else {
                        std::cout << " " << feats[i][d];
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            auto fst_order = fst::topo_order(frame_fst);
            fst::forward_one_best<ilat::fst> one_best;

            for (auto& i: frame_fst.initials()) {
                one_best.extra[i] = {-1, 0};
            }

            one_best.merge(frame_fst, fst_order);
            std::vector<int> edges = one_best.best_path(frame_fst);

            if (ebt::in(std::string("frame-pred"), args)) {
                for (auto& e: edges) {
                    std::cout << i_args.id_label[frame_fst.output(e)] << " ";
                }
            } else {
                int prev = -1;

                for (auto& e: edges) {
                    if (i_args.id_label[frame_fst.output(e)] == "<blk>") {
                        prev = frame_fst.output(e);
                        continue;
                    }

                    if (frame_fst.output(e) != prev) {
                        std::cout << i_args.id_label[frame_fst.output(e)] << " ";
                    }

                    prev = frame_fst.output(e);
                }
            }

            std::cout << "(" << i << ".dot)" << std::endl;
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

}
