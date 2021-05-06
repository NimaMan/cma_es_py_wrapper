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


#include <iostream>
#include <vector>

#include "cmaes.h"

#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
//using namespace std;


PYBIND11_MODULE(es, m) 
{
    //declare_class_template_cma<CMAStrategy<CovarianceUpdate>, CMAParameters>(m);
    
    //template class CMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy>>;
    /* I need custom constructors and hence possibly lambda. 
        if I am gonna use lambda I need to give make lambda function recognize the tempalte classses 
    */
    py::class_< CMAES> (m, "cma")  
    /**
       * \brief Constructor.
       * @param dim problem dimensions
       * @param x0 initial search point
       * @param sigma initial distribution step size (positive, otherwise automatically set)
       * @param lambda number of offsprings sampled at each step
       * @param seed initial random seed, useful for reproducing results (if unspecified, automatically generated from current time)
       * @param gp genotype / phenotype object
       * @param sep whether to use sep-CMA-ES, using diagonal covariance matrix (modifies covariance default learning rate)
       */
      
    .def(py::init([](
            const int dim,
		    const double *x0,
		    const double sigma,
		    const int lambda=-1,
		    const uint64_t seed=1234){
            libcmaes::CMAParameters<> cmaparams(dim, x0, sigma, lambda, seed); 
            return new CMAES(cmaparams);
        }
        ), py::arg("dim"), py::arg("x0")=1, py::arg("sigma")=.1, py::arg("population_size")=-1, py::arg("seed")=1234 
        )
     /**
       * \brief Constructor.
       * @param x0 initial search point as vector of problem dimension
       * @param sigma initial distribution step size (positive, otherwise automatically set)
       * @param lambda number of offsprings sampled at each step
       * @param seed initial random seed, useful for reproducing results (if unspecified, automatically generated from current time)
       */
    .def(py::init([](
            const std::vector<double> x0,
		    const double sigma,
		    const int lambda=-1,
		    const uint64_t seed=1234){
            libcmaes::CMAParameters<> cmaparams(x0, sigma, lambda, seed); 
            return new CMAES(cmaparams);
        }
        ), py::arg("x0"), py::arg("sigma"), py::arg("population_size")=-1, py::arg("seed")=1234 
        )
    .def("ask", &CMAES::ask) 
    .def("tell", &CMAES::tell) 
    ;

}  

