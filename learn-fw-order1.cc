#include "seg/iscrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include <fstream>

namespace frank_wolfe {

    struct primal_t {
        std::vector<scrf::dense_vec> param;
        std::vector<double> cost;
    };

    primal_t load_primal(std::istream& is);
    primal_t load_primal(std::string filename);
    void save_primal(primal_t const& d, std::ostream& os);
    void save_primal(primal_t const& d, std::string filename);

}

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gold_batch;

    std::ifstream lattice_batch;

    std::string output_param;
    std::string output_opt_data;

    double l2;
    double samples;
    frank_wolfe::primal_t primal;
    std::string output_primal;

    iscrf::learning_args l_args;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-first",
        "Learn segmental CRF",
        {
            {"frame-batch", "", true},
            {"gold-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"primal", "", true},
            {"l2", "", true},
            {"samples", "", true},
            {"features", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-primal", "", false},
            {"loss", "", true},
            {"cost-scale", "", false},
            {"label", "", true},
            {"use-gold-segs", "", false},
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
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    gold_batch.open(args.at("gold-batch"));

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
    }

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    l2 = std::stod(args.at("l2"));
    samples = std::stoi(args.at("samples"));

    primal = frank_wolfe::load_primal(args.at("primal"));

    output_primal = "primal-last";
    if (ebt::in(std::string("output-primal"), args)) {
        output_primal = args.at("output-primal");
    }

    iscrf::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    ebt::Timer timer;

    int sample = 0;

    while (1) {

        iscrf::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        s.gold_segs = util::load_segments(gold_batch, l_args.label_id);

        if (!gold_batch) {
            break;
        }

        if (ebt::in(std::string("lattice-batch"), args)) {
            ilat::fst lat = ilat::load_lattice(lattice_batch, l_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            iscrf::make_lattice(lat, s, l_args);
        } else {
            iscrf::make_graph(s, l_args);
        }

        iscrf::make_min_cost_gold(s, l_args);

        parameterize(s, l_args);

        double gold_cost = 0;

        iscrf::iscrf_fst gold { s.gold_data };
    
        std::cout << "gold path: ";
        for (auto& e: gold.edges()) {
            std::cout << l_args.id_label[gold.output(e)] << " ";
            gold_cost += iscrf::cost(s.gold_data, e);
        }
        std::cout << std::endl;
    
        std::cout << "gold cost: " << gold_cost << std::endl;

        std::shared_ptr<scrf::loss_func<scrf::dense_vec>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s.graph_data.cost_func);
            s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            loss_func = std::make_shared<hinge_loss>(hinge_loss { s.gold_data, s.graph_data });

            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            iscrf::iscrf_fst gold { s.gold_data };

            std::cout << "gold: ";
            for (auto& e: gold.edges()) {
                std::cout << l_args.id_label[gold.output(e)] << " ";
                gold_weight += gold.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            iscrf::iscrf_fst graph_path { loss.graph_path };

            std::cout << "cost aug: ";
            for (auto& e: graph_path.edges()) {
                std::cout << l_args.id_label[graph_path.output(e)] << " ";
                graph_weight += graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl;
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << s.gold_data.fst->edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        if (ell < 0) {
            std::cout << "loss is less than zero" << std::endl;
            exit(1);
        } else {
            scrf::dense_vec param_grad = loss_func->param_grad();

            imul(param_grad, -1 / (l2 * samples));

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;
            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());
            iscrf::iscrf_fst graph_path { loss.graph_path };
            double cost = 0;

            for (auto& e: graph_path.edges()) {
                cost += (*s.graph_data.cost_func)(*s.graph_data.fst, e);
            }

            scrf::dense_vec d = primal.param[sample];
            isub(d, param_grad);

            double gamma = (l2 * scrf::dot(d, l_args.param) - primal.cost[sample] + cost / samples)
                / (l2 * scrf::dot(d, d));

            std::cout << "gamma: " << gamma << std::endl;

            gamma = std::min(1.0, std::max(0.0, gamma));

            std::cout << "gamma clipped: " << gamma << std::endl;

            scrf::dense_vec update = primal.param[sample];
            scrf::dense_vec tmp = param_grad;
            imul(update, 1 - gamma);
            imul(tmp, gamma);
            iadd(update, tmp);

            primal.cost[sample] = (1 - gamma) * primal.cost[sample] + gamma * cost / samples;
            isub(l_args.param, primal.param[sample]);
            iadd(l_args.param, update);
            primal.param[sample] = update;
        }

        std::cout << std::endl;

#if DEBUG_TOP
        if (sample == DEBUG_TOP) {
            break;
        }
#endif

        ++sample;
    }

    scrf::save_vec(l_args.param, output_param);
    scrf::save_vec(l_args.opt_data, output_opt_data);

    frank_wolfe::save_primal(primal, output_primal);

}


namespace frank_wolfe {

    primal_t load_primal(std::istream& is)
    {
        primal_t d;

        std::string line;
        std::getline(is, line);

        int samples = std::stoi(line);

        for (int i = 0; i < samples; ++i) {
            d.param.push_back(scrf::load_dense_vec(is));
            std::getline(is, line);
            d.cost.push_back(std::stod(line));
        }

        return d;
    }

    primal_t load_primal(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_primal(ifs);
    }

    void save_primal(primal_t const& d, std::ostream& os)
    {
        os << d.param.size() << std::endl;

        for (int i = 0; i < d.param.size(); ++i) {
            scrf::save_vec(d.param[i], os);
            os << d.cost[i] << std::endl;
        }
    }

    void save_primal(primal_t const& d, std::string filename)
    {
        std::ofstream ofs { filename };
        save_primal(d, ofs);
    }

}
