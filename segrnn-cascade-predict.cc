#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "segbin/cascade.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    int min_seg;
    int max_seg;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;

    double alpha;

    std::vector<std::string> pass1_features;
    std::vector<std::string> pass2_features;
    std::shared_ptr<tensor_tree::vertex> param;

    int outer_layer;
    int inner_layer;
    std::shared_ptr<tensor_tree::vertex> nn_param;

    std::shared_ptr<ilat::fst> lm;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-cascade-predict",
        "Decode with segmental RNN",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", true},
            {"pass1-features", "", true},
            {"pass2-features", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"alpha", "", true},
            {"lm", "", false},
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
    frame_batch.open(args.at("frame-batch"));

    std::tie(outer_layer, inner_layer, nn_param, std::ignore)
        = fscrf::load_lstm_param(args.at("nn-param"));

    alpha = std::stod(args.at("alpha"));

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    pass1_features = ebt::split(args.at("pass1-features"), ",");
    pass2_features = ebt::split(args.at("pass2-features"), ",");

    param = cascade::make_tensor_tree(pass1_features, pass2_features);

    tensor_tree::load_tensor(param, args.at("param"));

    label_id = util::load_label_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }

    std::ifstream lm_stream { args.at("lm") };
    lm = std::make_shared<ilat::fst>(ilat::load_arpa_lm(lm_stream, label_id));
    lm_stream.close();
}

void prediction_env::run()
{
    int nsample = 0;

    while (1) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        lstm_var_tree = make_var_tree(comp_graph, nn_param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(frames[i])));
        }

        std::shared_ptr<lstm::transcriber> trans = fscrf::make_transcriber(outer_layer, inner_layer, args, nullptr);

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops = (*trans)(lstm_var_tree, frame_ops);

        fscrf::fscrf_data graph_data;

        graph_data.fst = fscrf::make_graph(feat_ops.size(),
            label_id, id_label, min_seg, max_seg, 1);
        graph_data.topo_order = std::make_shared<std::vector<int>>(
            ::fst::topo_order(*graph_data.fst));

        auto frame_mat = autodiff::row_cat(feat_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        graph_data.weight_func = fscrf::make_weights(pass1_features, var_tree->children[0], frame_mat);

        fscrf::fscrf_fst graph { graph_data };

        cascade::cascade cas { graph };

        cas.compute_marginal();

        double inf = std::numeric_limits<double>::infinity();
        int reachable_edges = 0;
        double sum = 0;
        double max = -inf;

        for (int e = 0; e < cas.max_marginal.size(); ++e) {
            sum += cas.max_marginal[e];

            if (cas.max_marginal[e] > max) {
                max = cas.max_marginal[e];
            }

            ++reachable_edges;
        }

        double threshold = alpha * max + (1 - alpha) * sum / reachable_edges;

        ilat::fst_data lat_data;
        std::unordered_map<int, int> edge_map;

        std::tie(lat_data, edge_map) = cas.compute_lattice(threshold, label_id, id_label);
        ilat::fst lat;
        lat.data = std::make_shared<ilat::fst_data>(lat_data);

        ilat::add_eps_loops(lat);
        ilat::lazy_pair_mode1 composed_fst { lat, *lm };
        fscrf::fscrf_pair_data pair_graph_data;
        pair_graph_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(fst::topo_order(composed_fst));
        pair_graph_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);

        std::shared_ptr<scrf::composite_weight<ilat::pair_fst>> pass2_weights
            = fscrf::make_pair_weights(pass2_features, var_tree->children[1], frames);

        pass2_weights->weights.push_back(
            std::make_shared<fscrf::mode1_weight>(fscrf::mode1_weight {
                std::make_shared<fscrf::pass_through_score>(
                fscrf::pass_through_score { tensor_tree::get_var(var_tree->children[1]->children.back()),
                graph_data.weight_func, *graph_data.fst, edge_map })
            })
        );

        pair_graph_data.weight_func = pass2_weights;

        fscrf::fscrf_pair_data graph_path_data;
        graph_path_data.fst = scrf::shortest_path(pair_graph_data);

        fscrf::fscrf_pair_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            if (graph_path.output(e) == 0) {
                continue;
            }

            std::cout << id_label.at(graph_path.output(e)) << " ";
        }
        std::cout << "(" << nsample << ".dot)";
        std::cout << std::endl;

        ++nsample;
    }

}
