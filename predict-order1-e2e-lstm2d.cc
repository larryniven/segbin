#include "seg/iscrf_e2e.h"
#include "seg/iscrf_e2e_lstm2d.h"
#include "seg/loss.h"
#include "seg/scrf_weight.h"
#include "autodiff/autodiff.h"
#include "nn/lstm.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    std::ifstream lattice_batch;

    iscrf::e2e_lstm2d::inference_args i_args;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-order1-e2e",
        "Decode with segmental CRF",
        {
            {"frame-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"stride", "", false},
            {"param", "", true},
            {"nn-param", "", true},
            {"features", "", true},
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

    iscrf::e2e_lstm2d::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int i = 1;

    while (1) {

        iscrf::sample s { i_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;

        std::vector<std::shared_ptr<autodiff::op_t>> inputs;
        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(comp_graph.var(la::vector<double>(frames[i])));
        }

        lstm::db_lstm2d_nn_t nn = lstm::make_db_lstm2d_nn(comp_graph, i_args.nn_param, inputs);

        auto topo_order = autodiff::topo_order(nn.layer.back().output);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        std::vector<std::vector<double>> outputs;
        for (int i = 0; i < nn.layer.back().output.size(); ++i) {
            auto& v = autodiff::get_output<la::vector<double>>(nn.layer.back().output[i]);
            outputs.push_back(std::vector<double> { v.data(), v.data() + v.size() });
        }

        s.frames = outputs;

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

        iscrf::parameterize(s.graph_data, s.graph_alloc, s.frames, i_args);

        std::shared_ptr<ilat::fst> graph_path = scrf::shortest_path<iscrf::iscrf_data>(s.graph_data);

        for (auto& e: graph_path->edges()) {
            std::cout << i_args.id_label.at(graph_path->output(e)) << " ";
        }
        std::cout << "(" << i << ".phn)" << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

}

