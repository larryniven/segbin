#include "seg/pair_scrf.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "seg/iscrf.h"
#include "seg/util.h"
#include <fstream>

namespace second_order = scrf::second_order;

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gold_batch;

    std::ifstream lattice_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    second_order::learning_args l_args;

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
            {"frame-batch", "", false},
            {"lm", "", true},
            {"gold-batch", "", true},
            {"lattice-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"loss", "", true},
            {"label", "", true},
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

    save_every = std::numeric_limits<int>::max();
    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    }

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    l_args = second_order::parse_learning_args(args);
}

void learning_env::run()
{
    int i = 1;

    while (1) {

        second_order::learning_sample s { l_args };

        if (ebt::in(std::string("frame-batch"), args)) {
            s.frames = speech::load_frame_batch(frame_batch);
        }

        s.gold_segs = scrf::load_segments(gold_batch, l_args.label_id);

        if (!gold_batch) {
            break;
        }

        ilat::fst lat = ilat::load_lattice(lattice_batch, l_args.label_id);

        if (!lattice_batch) {
            std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
            exit(1);
        }

        second_order::make_lattice(lat, s, l_args);

        second_order::make_min_cost_gold(s, l_args);

        s.cost = std::make_shared<scrf::seg_cost<ilat::pair_fst>>(
            scrf::make_overlap_cost<ilat::pair_fst>(s.gold_segs, l_args.sils));

        std::shared_ptr<scrf::loss_func<scrf::sparse_vec>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::pair_fst>& graph_weight_func
                = *dynamic_cast<scrf::composite_weight<ilat::pair_fst>*>(s.graph.weight_func.get());
            graph_weight_func.weights.push_back(s.cost);

            using hinge_loss = scrf::hinge_loss<
                second_order::pair_scrf, scrf::sparse_vec,
                second_order::pair_scrf_path_maker>;

            loss_func = std::make_shared<hinge_loss>(hinge_loss { *s.gold, s.graph });

            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: s.gold->edges()) {
                std::cout << l_args.id_label[s.gold->output(e)] << " ";
                gold_weight += s.gold->weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            std::cout << "cost aug: ";
            for (auto& e: loss.graph_path->edges()) {
                std::cout << l_args.id_label[loss.graph.output(e)] << " ";
                graph_weight += loss.graph_path->weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl; 
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << s.gold->edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::sparse_vec param_grad;

        if (ell > 0) {
            param_grad = loss_func->param_grad();
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data, l_args.step_size);

        if (i % save_every == 0) {
            scrf::save_vec(l_args.param, "param-last");
            scrf::save_vec(l_args.opt_data, "opt-data-last");
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_vec(l_args.param, output_param);
    scrf::save_vec(l_args.opt_data, output_opt_data);

}

