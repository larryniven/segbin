#include "segbin/cascade.h"

namespace cascade {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
        std::vector<std::string> const& pass1_features,
        std::vector<std::string> const& pass2_features)
    {
        tensor_tree::vertex root;
    
        root.children.push_back(fscrf::make_tensor_tree(pass1_features));
        root.children.push_back(fscrf::make_tensor_tree(pass2_features));
    
        return std::make_shared<tensor_tree::vertex>(root);
    }

    cascade::cascade(fscrf::fscrf_fst& graph)
        : graph(graph)
    {}
    
    void cascade::compute_marginal()
    {
        for (auto v: graph.initials()) {
            forward.extra[v] = {-1, 0};
        }
        forward.merge(graph, graph.topo_order());
    
        for (auto v: graph.finals()) {
            backward.extra[v] = {-1, 0};
        }
        backward.merge(graph, graph.topo_order());
    
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
    
        std::vector<int> edges = graph.edges();
    
        max_marginal.resize(edges.size());
    
        for (int e: edges) {
            auto tail = graph.tail(e);
            auto head = graph.head(e);
    
            int tail_time = graph.time(tail);
            int head_time = graph.time(head);
    
            max_marginal[e] = fb_alpha(tail) + graph.weight(e) + fb_beta(head);
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
    }
    
    std::tuple<ilat::fst_data, std::unordered_map<int, int>> cascade::compute_lattice(
        double threshold, std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ilat::fst_data data;
        std::unordered_map<int, int> inv_edge_map;
    
        std::unordered_map<int, int> vertex_map;
    
        std::vector<int> stack;
        std::unordered_set<int> traversed;
    
        for (auto v: graph.initials()) {
            stack.push_back(v);
            traversed.insert(v);
        }
    
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);
    
        while (stack.size() > 0) {
            auto u = stack.back();
            stack.pop_back();
    
            for (auto&& e: graph.out_edges(u)) {
                auto tail = graph.tail(e);
                auto head = graph.head(e);
    
                double weight = graph.weight(e);
    
                if (max_marginal[e] > threshold) {
    
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
                    inv_edge_map[e_new] = e;
    
                    if (!ebt::in(head, traversed)) {
                        stack.push_back(head);
                        traversed.insert(head);
                    }
    
                }
            }
        }
    
        for (auto& i: graph.initials()) {
            data.initials.push_back(vertex_map.at(i));
        }
    
        for (auto& f: graph.finals()) {
            data.finals.push_back(vertex_map.at(f));
        }
    
        return std::make_tuple(data, inv_edge_map);
    }

}
