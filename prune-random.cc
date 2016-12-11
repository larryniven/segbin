#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>
#include <random>

struct pruning_env {

    std::ifstream frame_batch;

    fscrf::inference_args i_args;

    double prob;
    int seed;

    std::ofstream output;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "prune-random",
        "Prune randomly",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"label", "", true},
            {"prob", "", true},
            {"output", "", true},
            {"seed", "", true},
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

    prob = std::stod(args.at("prob"));

    seed = std::stoi(args.at("seed"));

    output.open(args.at("output"));

    fscrf::parse_inference_args(i_args, args);
}

void pruning_env::run()
{
    std::default_random_engine gen { seed };
    std::bernoulli_distribution dist { prob };

    int i = 1;

    while (1) {

        fscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        fscrf::make_graph(s, i_args);

        fscrf::fscrf_fst graph { s.graph_data };
        s.graph_data.topo_order = std::make_shared<std::vector<int>>(
            fst::topo_order(graph));

        std::cout << "sample: " << i << std::endl;
        std::cout << "frames: " << s.frames.size() << std::endl;

        std::unordered_map<int, int> vertex_map;

        ilat::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(i_args.label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(i_args.id_label);

        for (int v: graph.topo_order()) {
            for (auto& e: graph.out_edges(v)) {
                bool drop = dist(gen);

                if (!drop) {
                    auto tail = graph.tail(e);
                    auto head = graph.head(e);

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
                    ilat::add_edge(data, e_new, ilat::edge_data { tail_new, head_new, 0,
                        graph.input(e), graph.output(e) });
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

        auto edges = graph.edges();

        std::cout << "edges: " << edges.size() << " left: " << f.edges().size()
            << " (" << double(f.edges().size()) / edges.size() << ")" << std::endl;

        std::cout << std::endl;

        ++i;
    }
}

