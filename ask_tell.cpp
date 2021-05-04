/**
 * CMA-ES, Covariance Matrix Adaptation Evolution Strategy
 * Copyright (c) 2014 Inria
 * Author: Emmanuel Benazera <emmanuel.benazera@lri.fr>
 *
 * This file is part of libcmaes.
 *
 * libcmaes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libcmaes is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcmaes.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "src/cmaes.h"
#include <iostream>
#include <vector>
using namespace libcmaes;


FitFunc fsphere = [](const double *x, const int N)
{
  double val = 0.0;
  for (int i=0;i<N;i++)
    val += x[i]*x[i];
  return val;
};


std::vector<double> eval(const dMat &candidates){
    // custom eval.
    std::vector<double> fvals(candidates.cols());
    for (int r=0;r<candidates.cols();r++){
        fvals[r] = fsphere(candidates.col(r).data(), candidates.rows());
	      //std::cerr << "candidate x: " << _solutions.get_candidate(r).get_x_dvec().transpose() << std::endl;
    }
    //update_fevals(candidates.cols());
    return fvals;
};
  

int main(int argc, char *argv[])
{
  int dim = 10; // problem dimensions.
  std::vector<double> x0(dim,10.0);
  double sigma = 0.1;

  CMAParameters<> cmaparams(x0,sigma);
  //ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters<>> optim(fsphere,cmaparams);
  ESOptimizer<CMAStrategy<CovarianceUpdate>, CMAParameters<>> optim(cmaparams);
  
  for (int i=0; i<500; i++){
      dMat candidates = optim.ask();
      std::vector<double> fvals = eval(candidates);
      optim.tell(fvals);
    }
  std::cout << optim.get_solutions() << std::endl;
}


/*
PYBIND11_MODULE(cmapy, m) {
    py::class_<TSPBRKGA>(m, "TSPBRKGA")
	.def(py::init([](vector<vector<float>> nodeCoords, int n, int p, double pe, double pm, double rhoe, int K, int MAX_THREADS, uint32_t seed) {
                 return new TSPBRKGA(nodeCoords, n, p, pe, pm, rhoe, K, MAX_THREADS, seed); 
                 }
                )
        )
    .def("evolve", &BRKGA::evolve)
    .def("get_best_fitness", &BRKGA::getBestFitness)
    .def("run_brkga", &TSPBRKGA::runBRKGA, py::arg("max_gens")=1000, py::arg("exchange_number")=2, py::arg("exchange_interval")=100, py::arg("verbose")=false)
    .def_readonly("get_fitness_progress", &BRKGA::fitnessProgress)
    ;
}

*/