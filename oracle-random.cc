#include "seg/seg-util.h"
#include "fst/ifst.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "fst/fst-algo.h"
#include <fstream>

struct oracle_env {

    std::ifstream label_batch;
    std::ifstream frame_batch;

    int max_seg;
    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;

    int seed;
    double drop_prob;

    std::vector<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    oracle_env(std::unordered_map<std::string, std::string> args);

    void run();

};

ifst::fst make_label_fst(std::vector<std::string> const& label_seq,
    std::unordered_map<std::string, int> const& label_id,
    std::vector<int> const& labels);

std::tuple<int, int, int, int> error_analysis(std::vector<std::tuple<int, int>> const& edges,
    fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst> const& composed_fst);

std::shared_ptr<ifst::fst> drop_edges(
    std::shared_ptr<ifst::fst> graph, std::default_random_engine& gen, double drop_prob);

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "oracle-random",
        "Calculate orracle error rate of random lattice",
        {
            {"label-batch", "", true},
            {"frame-batch", "", true},
            {"label", "", true},
            {"max-seg", "", false},
            {"print-path", "", false},
            {"ignore", "", false},
            {"seed", "", false},
            {"drop-prob", "", false},
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

    oracle_env env { args };

    env.run();

    return 0;
}

oracle_env::oracle_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    label_batch.open(args.at("label-batch"));
    frame_batch.open(args.at("frame-batch"));

    label_id = speech::load_label_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }

    if (ebt::in(std::string("ignore"), args)) {
        ignored = ebt::split(args.at("ignore"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    drop_prob = 0;
    if (ebt::in(std::string("drop-prob"), args)) {
        drop_prob = std::stod(args.at("drop-prob"));
    }
}

void oracle_env::run()
{
    ebt::Timer timer;

    int i = 1;

    int total_len = 0;

    int total_ins = 0;
    int total_del = 0;
    int total_sub = 0;

    double total_density = 0;

    std::default_random_engine gen { seed };

    while (1) {

        std::vector<std::string> label_seq = speech::load_label_seq_batch(label_batch);
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!label_batch || !frame_batch) {
            break;
        }

        std::shared_ptr<ifst::fst> graph = seg::make_graph(frames.size(), label_id, id_label,
            1, max_seg, 1);

        std::shared_ptr<ifst::fst> lat = drop_edges(graph, gen, drop_prob);

        ifst::add_eps_loops(*lat);

        ifst::fst label_fst = make_label_fst(label_seq, label_id, labels);

        for (auto& ig: ignored) {
            ifst::add_eps_loops(label_fst, label_id.at(ig));
        }

        fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst> composed_fst { *lat, label_fst };

        std::vector<std::tuple<int, int>> topo_order;

        for (int i = 0; i < lat->vertices().size(); ++i) {
            for (int j = 0; j < label_fst.vertices().size(); ++j) {
                topo_order.push_back(std::make_tuple(i, j));
            }
        }
        // auto topo_order = fst::topo_order(composed_fst);

        fst::forward_one_best<fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst>> one_best;
        for (auto& i: composed_fst.initials()) {
            one_best.extra[i] = { std::make_tuple(-1, -1), 0 };
        }
        one_best.merge(composed_fst, topo_order);
        std::vector<std::tuple<int, int>> best_edges = one_best.best_path(composed_fst);

        if (ebt::in(std::string("print-path"), args)) {
            std::cout << lat->data->name << std::endl;

            for (auto& e: best_edges) {
                int tail = std::get<0>(composed_fst.tail(e));
                int head = std::get<0>(composed_fst.head(e));

                if (tail == head) {
                    continue;
                }

                std::cout << lat->time(tail) << " " << lat->time(head)
                    << " " << id_label.at(composed_fst.input(e))
                    << std::endl;
            }

            std::cout << "." << std::endl;

        } else {
            int ins = 0;
            int del = 0;
            int sub = 0;
            int length = 0;

            std::tie(ins, del, sub, length) = error_analysis(best_edges, composed_fst);

            std::cout << lat->data->name << ": edges: " << lat->edges().size()
                << " density: " << lat->edges().size() / length << std::endl;

            std::cout << "ins: " << ins << " del: " << del << " sub: " << sub << " len: " << length
                << " er: " << double(ins + del + sub) / length << std::endl;

            total_ins += ins;
            total_del += del;
            total_sub += sub;
            total_len += length;

            total_density += lat->edges().size() / length;
        }

        ++i;
    }

    if (!ebt::in(std::string("print-path"), args)) {
        std::cout << "total ins: " << total_ins
            << " total del: " << total_del
            << " total sub: " << total_sub
            << " total len: " << total_len
            << " er: " << double(total_ins + total_del + total_sub) / total_len << std::endl;
        std::cout << "avg density: " << total_density / (i - 1) << std::endl;
    }
}

ifst::fst make_label_fst(std::vector<std::string> const& label_seq,
    std::unordered_map<std::string, int> const& label_id,
    std::vector<int> const& labels)
{
    ifst::fst_data data;
    data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);

    int v = 0;
    ifst::add_vertex(data, v, ifst::vertex_data { v });

    for (int i = 0; i < label_seq.size(); ++i) {
        int u = data.vertices.size();
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        // substitution
        for (int ell: labels) {
            if (ell == 0) {
                continue;
            }

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v, u,
                (ell == label_id.at(label_seq.at(i)) ? 0.0 : -1.0),
                ell, label_id.at(label_seq.at(i)) });
        }

        // insertion
        int e = data.edges.size();
        ifst::add_edge(data, e, ifst::edge_data { v, u,
            -1.0, 0, label_id.at(label_seq.at(i)) });

        v = u;
    }

    data.initials.push_back(0);
    data.finals.push_back(v);

    for (int v = 0; v < data.vertices.size(); ++v) {
        // deletion
        for (int ell: labels) {
            if (ell == 0) {
                continue;
            }

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v, v,
                -1, ell, 0 });
        }
    }

    ifst::fst f;
    f.data = std::make_shared<ifst::fst_data>(data);

    return f;
}

std::tuple<int, int, int, int> error_analysis(std::vector<std::tuple<int, int>> const& edges,
    fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst> const& composed_fst)
{
    int ins = 0;
    int del = 0;
    int sub = 0;
    int length = 0;

    for (auto& e: edges) {
        if (composed_fst.input(e) == 0) {
            ++del;
        } else if (composed_fst.output(e) == 0) {
            ++ins;
        } else if (composed_fst.input(e) != composed_fst.output(e)) {
            ++sub;
        }

        if (composed_fst.output(e) != 0) {
            ++length;
        }
    }

    return std::make_tuple(ins, del, sub, length);

}

std::shared_ptr<ifst::fst> drop_edges(
    std::shared_ptr<ifst::fst> graph, std::default_random_engine& gen, double drop_prob)
{
    std::bernoulli_distribution dist { drop_prob };

    ifst::fst_data data;

    for (int i: graph->vertices()) {
        ifst::add_vertex(data, i, { graph->time(i) });
    }

    for (int e: graph->edges()) {
        if (dist(gen) == 1) {
            continue;
        }

        ifst::add_edge(data, e, { graph->tail(e), graph->head(e), 0, graph->input(e), graph->output(e) });
    }

    for (int i: graph->initials()) {
        data.initials.push_back(i);
    }

    for (int f: graph->finals()) {
        data.finals.push_back(f);
    }

    return std::make_shared<ifst::fst>(ifst::fst { std::make_shared<ifst::fst_data>(data) });
}
