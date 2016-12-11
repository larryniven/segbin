#ifndef CASCADE_H
#define CASCADE_H

#include "nn/tensor-tree.h"
#include "seg/fscrf.h"

namespace cascade {

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
         std::vector<std::string> const& pass1_features,
         std::vector<std::string> const& pass2_features);

    struct cascade {
    
        fscrf::fscrf_fst& graph;
    
        cascade(fscrf::fscrf_fst& graph);
    
        fst::forward_one_best<fscrf::fscrf_fst> forward;
        fst::backward_one_best<fscrf::fscrf_fst> backward;
    
        std::vector<double> max_marginal;
    
        void compute_marginal();
    
        std::tuple<ilat::fst_data, std::unordered_map<int, int>> compute_lattice(
            double threshold, std::unordered_map<std::string, int> const& label_id,
            std::vector<std::string> const& id_label);
    
    };

}

#endif
