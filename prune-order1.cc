#include "seg/iscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "autodiff/autodiff.h"
#include "nn/lstm.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    std::ifstream lattice_batch;

    std::ofstream output;

    iscrf::inference_args i_args;

    double alpha;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-order1",
        "Decode with segmental CRF",
        {
            {"frame-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
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

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
    }

    iscrf::parse_inference_args(i_args, args);

    alpha = std::stod(args.at("alpha"));

    output.open(args.at("output"));
}

void prediction_env::run()
{
    int i = 1;

    while (1) {

        iscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        if (ebt::in(std::string("lattice-batch"), args)) {
            ilat::fst lat = ilat::load_lattice(lattice_batch, i_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            iscrf::make_lattice(lat, s, i_args);
        } else {
            iscrf::make_graph(s, i_args);
        }

        iscrf::parameterize_cached(s.graph_data, s.graph_alloc, s.frames, i_args);

        scrf::scrf_fst<iscrf::iscrf_data> graph { s.graph_data };
        auto order = graph.topo_order();

        fst::forward_one_best<scrf::scrf_fst<iscrf::iscrf_data>> forward;
        for (auto v: graph.initials()) {
            forward.extra[v] = {-1, 0};
        }
        forward.merge(graph, order);

        fst::backward_one_best<scrf::scrf_fst<iscrf::iscrf_data>> backward;
        for (auto v: graph.finals()) {
            backward.extra[v] = {-1, 0};
        }
        backward.merge(graph, order);

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

        std::cout << "frames: " << s.frames.size() << std::endl;
        std::cout << "max: " << max << " avg: " << sum / edge_count
            << " threshold: " << threshold << std::endl;
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

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

}

