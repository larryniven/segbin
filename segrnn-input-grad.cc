#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "speech/speech.h"
#include <fstream>

struct inspection_env {

    std::ifstream frame_batch;

    fscrf::inference_args i_args;

    int start_time;
    int end_time;
    std::string label;

    std::unordered_map<std::string, std::string> args;

    inspection_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-input-grad",
        "Compute input grad with segmental RNN",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"logsoftmax", "", false},
            {"edge", "", true},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    inspection_env env { args };

    env.run();

    return 0;
}

inspection_env::inspection_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    fscrf::parse_inference_args(i_args, args);

    std::vector<std::string> parts = ebt::split(args.at("edge"));

    assert(parts.size() == 3);

    start_time = std::stoi(parts[0]);
    end_time = std::stoi(parts[1]);
    label = parts[2];
}

void inspection_env::run()
{
    fscrf::sample s { i_args };

    s.frames = speech::load_frame_batch(frame_batch);

    autodiff::computation_graph comp_graph;
    std::shared_ptr<tensor_tree::vertex> var_tree
        = tensor_tree::make_var_tree(comp_graph, i_args.param);

    std::shared_ptr<tensor_tree::vertex> lstm_var_tree;

    lstm_var_tree = make_var_tree(comp_graph, i_args.nn_param);

    std::vector<std::shared_ptr<autodiff::op_t>> input;
    for (int i = 0; i < s.frames.size(); ++i) {
        auto f_var = comp_graph.var(la::tensor<double>(
            la::vector<double>(s.frames[i])));
        input.push_back(f_var);
    }

    std::vector<std::shared_ptr<autodiff::op_t>> frame_ops = input;

    std::shared_ptr<lstm::transcriber> trans = fscrf::make_transcriber(i_args);

    if (ebt::in(std::string("logsoftmax"), args)) {
        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { trans });
        frame_ops = (*trans)(lstm_var_tree, frame_ops);
    } else {
        frame_ops = (*trans)(lstm_var_tree->children[0], frame_ops);
    }

    fscrf::make_graph(s, i_args, frame_ops.size());

    auto frame_mat = autodiff::row_cat(frame_ops);

    autodiff::eval(frame_mat, autodiff::eval_funcs);

    s.graph_data.weight_func = fscrf::make_weights(i_args.features, var_tree, frame_mat);

    fscrf::fscrf_fst graph { s.graph_data };

    int target_edge = -1;

    for (auto& e: graph.edges()) {
        int tail = graph.tail(e);
        int head = graph.head(e);

        if (graph.time(tail) == start_time
                && graph.time(head) == end_time
                && graph.output(e) == i_args.label_id.at(label)) {
            target_edge = e;
            break;
        }
    }

    graph.weight(target_edge);

    s.graph_data.weight_func->accumulate_grad(1, *s.graph_data.fst, target_edge);

    s.graph_data.weight_func->grad();

    autodiff::grad(frame_mat, autodiff::grad_funcs);

    for (int j = 0; j < input.size(); ++j) {
        auto& t = autodiff::get_grad<la::tensor_like<double>>(input[j]);

        for (int i = 0; i < t.vec_size(); ++i) {
            std::cout << t({i});

            if (i != t.vec_size() - 1) {
                std::cout << " ";
            }
        }

        std::cout << std::endl;
    }

}

