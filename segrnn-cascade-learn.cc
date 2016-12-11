#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "segbin/cascade.h"
#include <fstream>

struct learning_env {

    util::batch_indices frame_batch;
    util::batch_indices label_batch;

    int min_seg;
    int max_seg;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;

    double clip;
    double step_size;

    double alpha;

    std::vector<std::string> pass1_features;
    std::vector<std::string> pass2_features;
    std::shared_ptr<tensor_tree::vertex> param;
    std::shared_ptr<tensor_tree::vertex> opt_data;

    int outer_layer;
    int inner_layer;
    std::shared_ptr<tensor_tree::vertex> nn_param;
    std::shared_ptr<tensor_tree::vertex> nn_opt_data;
    std::shared_ptr<tensor_tree::vertex> pred_param;
    std::shared_ptr<tensor_tree::vertex> pred_opt_data;

    std::string output_param;
    std::string output_opt_data;
    std::string output_nn_param;
    std::string output_nn_opt_data;

    std::default_random_engine gen;

    std::shared_ptr<ilat::fst> lm;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "segrnn-cascade",
        "Learn with segmental RNN",
        {
            {"frame-batch", "", false},
            {"label-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"pass1-features", "", true},
            {"pass2-features", "", true},
            {"label", "", true},
            {"subsampling", "", false},
            {"dropout", "", false},
            {"alpha", "", true},
            {"lm", "", false},
            {"step-size", "", true},
            {"clip", "", true},
            {"const-step-update", "", false},
            {"shuffle", "", false},
            {"seed", "", false},
            {"output-param", "", true},
            {"output-opt-data", "", true},
            {"output-nn-param", "", true},
            {"output-nn-opt-data", "", true},
            {"freeze-1st-pass", "", false}
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

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));
    label_batch.open(args.at("label-batch"));

    std::tie(outer_layer, inner_layer, nn_param, pred_param)
        = fscrf::load_lstm_param(args.at("nn-param"));
    std::tie(outer_layer, inner_layer, nn_opt_data, pred_opt_data)
        = fscrf::load_lstm_param(args.at("nn-opt-data"));

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
    opt_data = cascade::make_tensor_tree(pass1_features, pass2_features);

    tensor_tree::load_tensor(param, args.at("param"));
    tensor_tree::load_tensor(opt_data, args.at("opt-data"));

    label_id = util::load_label_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }

    if (ebt::in(std::string("seed"), args)) {
       gen = std::default_random_engine { std::stoul(args.at("seed")) };
    }

    std::ifstream lm_stream { args.at("lm") };
    lm = std::make_shared<ilat::fst>(ilat::load_arpa_lm(lm_stream, label_id));
    lm_stream.close();

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::vector<int> indices;
        indices.resize(frame_batch.pos.size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            frame_batch.pos[i] = pos[indices[i]];
        }

        pos = label_batch.pos;
        for (int i = 0; i < indices.size(); ++i) {
            label_batch.pos[i] = pos[indices[i]];
        }
    }

    output_param = args.at("output-param");
    output_opt_data = args.at("output-opt-data");
    output_nn_param = args.at("output-nn-param");
    output_nn_opt_data = args.at("output-nn-opt-data");
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_batch.pos.size()) {

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<int> label_seq = util::load_label_seq(label_batch.at(nsample), label_id);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<tensor_tree::vertex> lstm_var_tree;
        lstm_var_tree = make_var_tree(comp_graph, nn_param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(frames[i])));
        }

        std::shared_ptr<lstm::transcriber> trans = fscrf::make_transcriber(outer_layer, inner_layer, args, &gen);

        std::vector<std::shared_ptr<autodiff::op_t>> feat_ops = (*trans)(lstm_var_tree, frame_ops);

        fscrf::fscrf_data graph_data;

        graph_data.fst = fscrf::make_graph(feat_ops.size(),
            label_id, id_label, min_seg, max_seg, 1);
        graph_data.topo_order = std::make_shared<std::vector<int>>(
            ::fst::topo_order(*graph_data.fst));

        auto frame_mat = autodiff::row_cat(feat_ops);
        autodiff::eval(frame_mat, autodiff::eval_funcs);

        if (ebt::in(std::string("dropout"), args)) {
            graph_data.weight_func = fscrf::make_weights(pass1_features, var_tree->children[0], frame_mat,
                std::stod(args.at("dropout")), &gen);
        } else {
            graph_data.weight_func = fscrf::make_weights(pass1_features, var_tree->children[0], frame_mat);
        }

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

        std::cout << "mean: " << sum / reachable_edges << " threshold: " << threshold << std::endl;

        ilat::fst_data lat_data;
        std::unordered_map<int, int> edge_map;

        std::tie(lat_data, edge_map) = cas.compute_lattice(threshold,
            label_id, id_label);
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

        fscrf::marginal_log_loss_pair loss { pair_graph_data, label_seq };

        double ell = loss.loss();

        std::cout << "loss: " << ell << std::endl;

        std::shared_ptr<tensor_tree::vertex> param_grad = cascade::make_tensor_tree(pass1_features, pass2_features);
        std::shared_ptr<tensor_tree::vertex> nn_param_grad = fscrf::make_lstm_tensor_tree(outer_layer, inner_layer);

        if (ell > 0 && ebt::in(std::string("freeze-1st-pass"), args)) {
            loss.grad();

            pair_graph_data.weight_func->grad();
            tensor_tree::copy_grad(param_grad->children[1], var_tree->children[1]);

            if (ebt::in(std::string("clip"), args)) {
                double n = tensor_tree::norm(param_grad->children[1]);

                if (n > clip) {
                    tensor_tree::imul(param_grad->children[1], clip / n);

                    std::cout << "grad norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            if (ebt::in(std::string("const-step-update"), args)) {
                tensor_tree::const_step_update(param->children[1], param_grad->children[1], step_size);
            } else {
                tensor_tree::adagrad_update(param->children[1], param_grad->children[1], opt_data->children[1], step_size);
            }
        } else if (ell > 0) {
            loss.grad();

            pair_graph_data.weight_func->grad();
            tensor_tree::copy_grad(param_grad, var_tree);

            graph_data.weight_func->grad();
            autodiff::grad(frame_mat, autodiff::grad_funcs);
            tensor_tree::copy_grad(nn_param_grad, lstm_var_tree);

            if (ebt::in(std::string("clip"), args)) {
                double n1 = tensor_tree::norm(nn_param_grad);
                double n2 = tensor_tree::norm(param_grad);

                double n = std::sqrt(n1 * n1 + n2 * n2);

                if (n > clip) {
                    tensor_tree::imul(nn_param_grad, clip / n);
                    tensor_tree::imul(param_grad, clip / n);

                    std::cout << "grad norm: " << n << " clip: " << clip << " gradient clipped" << std::endl;
                }
            }

            double w1 = tensor_tree::get_matrix(nn_param->children[0]->children[0]->children[0]->children[0])(0, 0);

            if (ebt::in(std::string("const-step-update"), args)) {
                tensor_tree::const_step_update(nn_param, nn_param_grad, step_size);
                tensor_tree::const_step_update(param, param_grad, step_size);
            } else {
                tensor_tree::adagrad_update(nn_param, nn_param_grad, nn_opt_data, step_size);
                tensor_tree::adagrad_update(param, param_grad, opt_data, step_size);
            }

            double w2 = tensor_tree::get_matrix(nn_param->children[0]->children[0]->children[0]->children[0])(0, 0);
            std::cout << "weight: " << w1 << " update: " << w2 - w1 << " ratio: " << (w2 - w1) / w1 << std::endl;
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        double n1 = tensor_tree::norm(nn_param);
        double n2 = tensor_tree::norm(param);

        std::cout << "norm: " << std::sqrt(n1 * n1 + n2 * n2) << std::endl;
        std::cout << std::endl;

        ++nsample;
    }

    tensor_tree::save_tensor(param, output_param);
    fscrf::save_lstm_param(outer_layer, inner_layer,
        nn_param, pred_param, output_nn_param);

    tensor_tree::save_tensor(opt_data, output_opt_data);
    fscrf::save_lstm_param(outer_layer, inner_layer,
        nn_opt_data, pred_opt_data, output_nn_opt_data);

}

