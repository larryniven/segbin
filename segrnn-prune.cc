#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    fscrf::inference_args i_args;

    double alpha;

    std::ofstream output;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-predict",
        "Prune with segmental RNN",
        {
            {"frame-batch", "", false},
            {"label-batch", "", false},
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
            {"include-alignment", "", false}
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

    if (ebt::in(std::string("label-batch"), args)) {
        label_batch.open(args.at("label-batch"));
    }

    alpha = std::stod(args.at("alpha"));

    output.open(args.at("output"));

    fscrf::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int nsample = 1;

    while (1) {

        fscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> label_seq;

        if (ebt::in(std::string("label-batch"), args)) {
            label_seq = util::load_label_seq(label_batch);
        }

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree = make_var_tree(comp_graph, i_args.nn_param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::tensor<double>(la::vector<double>(s.frames[i]))));
        }

        if (ebt::in(std::string("nn-param"), args)) {
            std::shared_ptr<lstm::transcriber> trans
                = fscrf::make_transcriber(i_args);

            if (ebt::in(std::string("logsoftmax"), args)) {
                trans = std::make_shared<lstm::logsoftmax_transcriber>(
                    lstm::logsoftmax_transcriber { trans });
                frame_ops = (*trans)(lstm_var_tree, frame_ops);
            } else {
                frame_ops = (*trans)(lstm_var_tree->children[0], frame_ops);
            }
        }

        fscrf::make_graph(s, i_args, frame_ops.size());

        auto frame_mat = autodiff::row_cat(frame_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        s.graph_data.weight_func = fscrf::make_weights(i_args.features, var_tree, frame_mat);

        fscrf::fscrf_fst graph { s.graph_data };

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

        std::cout << "sample: " << nsample << std::endl;
        std::cout << "frames: " << s.frames.size() << std::endl;
        std::cout << "max: " << max << " avg: " << sum / edge_count
            << " alpha: " << alpha << " threshold: " << threshold << std::endl;
        std::cout << "forward: " << f_max << " backward: " << b_max << std::endl;

        std::vector<int> stack;
        std::unordered_set<int> traversed;

        for (auto v: graph.initials()) {
            stack.push_back(v);
            traversed.insert(v);
        }

        std::unordered_set<int> retained_edges;

        while (stack.size() > 0) {
            auto u = stack.back();
            stack.pop_back();

            for (auto&& e: graph.out_edges(u)) {
                auto tail = graph.tail(e);
                auto head = graph.head(e);

                double weight = graph.weight(e);

                if (fb_alpha(tail) + weight + fb_beta(head) > threshold) {

                    retained_edges.insert(e);

                    if (!ebt::in(head, traversed)) {
                        stack.push_back(head);
                        traversed.insert(head);
                    }

                }
            }
        }

        if (ebt::in(std::string("include-alignment"), args)) {
            std::vector<int> label_seq_id;
            for (auto& s: label_seq) {
                label_seq_id.push_back(i_args.label_id.at(s));
            }

            ilat::fst label_fst = fscrf::make_label_fst(label_seq_id, i_args.label_id, i_args.id_label);

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

            std::vector<std::tuple<int, int>> aligned_edges = one_best.best_path(pair);

            for (auto& e: aligned_edges) {
                retained_edges.insert(std::get<1>(e));
            }
        }

        ilat::fst_data data;
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
        }

        output << nsample << ".lat" << std::endl;

        ilat::fst f;
        f.data = std::make_shared<ilat::fst_data>(data);

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

        std::cout << "edges: " << edges.size() << " left: " << f.edges().size()
            << " (" << double(f.edges().size()) / edges.size() << ")" << std::endl;

        std::cout << std::endl;

        ++nsample;
    }
}

