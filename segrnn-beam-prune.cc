#include "seg/seg-util.h"
#include "speech/speech.h"
#include "fst/fst-algo.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    int inner_layer;
    int outer_layer;
    std::shared_ptr<tensor_tree::vertex> nn_param;

    seg::inference_args i_args;

    double alpha;

    std::ofstream output;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-beam-prune",
        "Prune with beam search",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", false},
            {"features", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"logsoftmax", "", false},
            {"alpha", "", false},
            {"output", "", true},
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
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    if (ebt::in(std::string("nn-param"), args)) {
        std::tie(outer_layer, inner_layer, nn_param)
            = seg::load_lstm_param(args.at("nn-param"));
    }

    alpha = std::stod(args.at("alpha"));

    output.open(args.at("output"));

    seg::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 1;

    while (1) {

        seg::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;

        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, nn_param);
        }

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

        seg::make_graph(s, i_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = seg::make_weights(i_args.features, var_tree, frame_mat);

        seg::seg_fst<seg::iseg_data> graph { s.graph_data };

        fst::beam_search<seg::seg_fst<seg::iseg_data>> beam_search;

        beam_search.merge(graph, *s.graph_data.topo_order, alpha);

        std::vector<int> retained_edges = beam_search.retained_edges;

        ifst::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(i_args.label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(i_args.id_label);

        std::unordered_map<int, int> vertex_map;

        for (auto& e: retained_edges) {
            int tail = graph.tail(e);
            int head = graph.head(e);
            double weight = graph.weight(e);

            if (!ebt::in(tail, vertex_map)) {
                int v = vertex_map.size();
                vertex_map[tail] = v;
                ifst::add_vertex(data, v, ifst::vertex_data { graph.time(tail) });
            }

            if (!ebt::in(head, vertex_map)) {
                int v = vertex_map.size();
                vertex_map[head] = v;
                ifst::add_vertex(data, v, ifst::vertex_data { graph.time(head) });
            }

            int tail_new = vertex_map.at(tail);
            int head_new = vertex_map.at(head);
            int e_new = data.edges.size();
            ifst::add_edge(data, e_new, ifst::edge_data { tail_new, head_new, weight,
                graph.input(e), graph.output(e) });
        }

        output << nsample << ".lat" << std::endl;

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        for (int i = 0; i < f.vertices().size(); ++i) {
            if (ebt::in(std::string("subsampling"), args)) {
                output << i << " "
                    << "time=" << f.time(i) * 4 << std::endl;
            } else {
                output << i << " "
                    << "time=" << f.time(i) << std::endl;
            }
        }

        output << "#" << std::endl;

        for (int e = 0; e < f.edges().size(); ++e) {
            int tail = f.tail(e);
            int head = f.head(e);

            output << tail << " " << head << " "
                << "label=" << i_args.id_label.at(f.output(e)) << ";"
                << "weight=" << f.weight(e) << std::endl;
        }
        output << "." << std::endl;

        auto edges = graph.edges();

        std::cout << "edges: " << edges.size() << " left: " << f.edges().size()
            << " (" << double(f.edges().size()) / edges.size() << ")" << std::endl;

        std::cout << std::endl;

        ++nsample;
    }
}

