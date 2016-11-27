#include "seg/fscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct pruning_env {

    std::ifstream frame_batch;

    fscrf::inference_args i_args;

    double alpha;

    std::ofstream output;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "prune-order1-full",
        "Prune with segmental CRF",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", false},
            {"features", "", true},
            {"label", "", true},
            {"alpha", "", true},
            {"output", "", true}
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

    pruning_env env { args };

    env.run();

    return 0;
}

pruning_env::pruning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    alpha = std::stod(args.at("alpha"));

    output.open(args.at("output"));

    fscrf::parse_inference_args(i_args, args);
}

void pruning_env::run()
{
    int i = 1;

    while (1) {

        fscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        fscrf::make_graph(s, i_args);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        std::shared_ptr<tensor_tree::vertex> pred_var_tree;
        if (ebt::in(std::string("nn-param"), args)) {
            lstm_var_tree = make_var_tree(comp_graph, i_args.nn_param);
            pred_var_tree = make_var_tree(comp_graph, i_args.pred_param);
        }

        lstm::stacked_bi_lstm_nn_t nn;
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

        std::shared_ptr<scrf::composite_weight<ilat::fst>> weight_func
            = fscrf::make_weights(i_args.features, var_tree, frame_mat);
        s.graph_data.weight_func = std::make_shared<
            scrf::cached_weight<ilat::fst>>(
            scrf::cached_weight<ilat::fst> { weight_func });

        fscrf::fscrf_fst graph { s.graph_data };
        s.graph_data.topo_order = std::make_shared<std::vector<int>>(
            fst::topo_order(graph));

        fst::forward_one_best<fscrf::fscrf_fst> forward;
        for (auto v: graph.initials()) {
            forward.extra[v] = {-1, 0};
        }
        forward.merge(graph, *s.graph_data.topo_order);

        fst::backward_one_best<fscrf::fscrf_fst> backward;
        for (auto v: graph.finals()) {
            backward.extra[v] = {-1, 0};
        }
        backward.merge(graph, *s.graph_data.topo_order);

        double inf = std::numeric_limits<double>::infinity();

        auto fb_alpha = [&](int v) {
            if (ebt::in(v, forward.extra)) {
                return forward.extra[v].value;
            } else {
                return -inf;
            }
        };

        auto fb_beta = [&](int v) {
            if (ebt::in(v, forward.extra)) {
                return backward.extra[v].value;
            } else {
                return -inf;
            }
        };

        double sum = 0;
        double max = -inf;

        auto edges = graph.edges();

        int edge_count = 0;

        for (auto& e: edges) {
            auto tail = graph.tail(e);
            auto head = graph.head(e);

            int tail_time = graph.time(tail);
            int head_time = graph.time(head);

            double s = fb_alpha(tail) + graph.weight(e) + fb_beta(head);

            if (s > max) {
                max = s;
            }

            if (s != -inf) {
                sum += s;
                ++edge_count;
            }
        }

        double b_max = -inf;

        for (auto& i: graph.initials()) {
            if (fb_beta(i) > b_max) {
                b_max = fb_beta(i);
            }
        }

        double f_max = -inf;

        for (auto& f: graph.finals()) {
            if (fb_alpha(f) > f_max) {
                f_max = fb_alpha(f);
            }
        }

        double threshold = alpha * max + (1 - alpha) * sum / edge_count;

        std::cout << "sample: " << i << std::endl;
        std::cout << "frames: " << s.frames.size() << std::endl;
        std::cout << "max: " << max << " avg: " << sum / edge_count
            << " alpha: " << alpha << " threshold: " << threshold << std::endl;
        std::cout << "forward: " << f_max << " backward: " << b_max << std::endl;

        std::unordered_map<int, int> vertex_map;

        std::vector<int> stack;
        std::unordered_set<int> traversed;

        for (auto v: graph.initials()) {
            stack.push_back(v);
            traversed.insert(v);
        }

        ilat::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(i_args.label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(i_args.id_label);

        while (stack.size() > 0) {
            auto u = stack.back();
            stack.pop_back();

            for (auto&& e: graph.out_edges(u)) {
                auto tail = graph.tail(e);
                auto head = graph.head(e);

                double weight = graph.weight(e);

                if (fb_alpha(tail) + weight + fb_beta(head) > threshold) {

                    if (!ebt::in(tail, vertex_map)) {
                        int v = vertex_map.size();
                        vertex_map[tail] = v;
                        ilat::add_vertex(data, v, ilat::vertex_data { graph.time(tail) });
                    }

                    if (!ebt::in(head, vertex_map)) {
                        int v = vertex_map.size();
                        vertex_map[head] = v;
                        ilat::add_vertex(data, v, ilat::vertex_data { graph.time(head) });
                    }

                    int tail_new = vertex_map.at(tail);
                    int head_new = vertex_map.at(head);
                    int e_new = data.edges.size();
                    ilat::add_edge(data, e_new, ilat::edge_data { tail_new, head_new, weight,
                        graph.input(e), graph.output(e) });

                    if (!ebt::in(head, traversed)) {
                        stack.push_back(head);
                        traversed.insert(head);
                    }

                }
            }
        }

        output << i << ".lat" << std::endl;

        ilat::fst f;
        f.data = std::make_shared<ilat::fst_data>(data);

        for (int i = 0; i < f.vertices().size(); ++i) {
            output << i << " "
                << "time=" << f.time(i) << std::endl;
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

        std::cout << "edges: " << edges.size() << " left: " << f.edges().size()
            << " (" << double(f.edges().size()) / edges.size() << ")" << std::endl;

        std::cout << std::endl;

        ++i;
    }
}

