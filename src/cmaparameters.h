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

#ifndef CMAPARAMETERS_H
#define CMAPARAMETERS_H

#include "parameters.h"
#include "eo_matrix.h"

#include <cmath>
#include <iostream>
#include <float.h>

namespace libcmaes
{
  /**
   * \brief Parameters for various flavors of the CMA-ES algorithm.
   */
  template <class TGenoPheno=GenoPheno<NoBoundStrategy> >
  class CMAParameters : public Parameters<TGenoPheno>
    {
      friend class CMASolutions;
      friend class CMAES;
      template <class U, class V> friend class CMAStrategy;
      template <class U, class V> friend class ESOStrategy;
      template <class U, class V> friend class IPOPCMAStrategy;
      template <class U, class V> friend class BIPOPCMAStrategy;
      //template <class U> friend class errstats;
      friend class CovarianceUpdate;
      friend class ACovarianceUpdate;
      template <class U> friend class errstats;
      friend class VDCMAUpdate;
      
    public:
      CMAParameters() {} //TODO: var init even if this constructor is not supposed to be used for now.
      
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
      CMAParameters(const int &dim,
		    const double &x0,
		    const double &sigma,
		    const int &lambda=-1,
		    const uint64_t &seed=0,
		    const TGenoPheno &gp=TGenoPheno())
    :Parameters<TGenoPheno>(dim,x0,lambda,seed,gp),_sigma_init(sigma),_nrestarts(9),_lazy_update(false),_lazy_value(0),_cm(1.0),_alphacov(2.0),_alphaminusold(0.5),_lambdamintarget(0.66),_alphaminusmin(1.0)
  {
    initialize_parameters();
  };
      
      /**
       * \brief Constructor.
       * @param x0 initial search point as vector of problem dimension
       * @param sigma initial distribution step size (positive, otherwise automatically set)
       * @param lambda number of offsprings sampled at each step
       * @param seed initial random seed, useful for reproducing results (if unspecified, automatically generated from current time)
       * @param gp genotype / phenotype object
       * @param sep whether to use sep-CMA-ES, using diagonal covariance matrix (modifies covariance default learning rate)
       */
      CMAParameters(const std::vector<double> &x0,
		    const double &sigma,
		    const int &lambda=-1,
		    const uint64_t &seed=0,
		    const TGenoPheno &gp=TGenoPheno())
    :Parameters<TGenoPheno>(x0.size(),&x0.front(),lambda,seed,gp),_sigma_init(sigma),_nrestarts(9),_lazy_update(false),_lazy_value(0),_cm(1.0),_alphacov(2.0),_alphaminusold(0.5),_lambdamintarget(0.66),_alphaminusmin(1.0)
  {
    initialize_parameters();
  };
      
      /**
       * \brief Constructor.
       * @param x0 initial search point as vector of problem dimension
       * @param sigma vector of initial distribution step sizes (positive, otherwise automatically set)
       * @param lambda number of offsprings sampled at each step
       * @param seed initial random seed, useful for reproducing results (if unspecified, automatically generated from current time)
       * @param gp genotype / phenotype object
       * @param sep whether to use sep-CMA-ES, using diagonal covariance matrix (modifies covariance default learning rate)
       */
      CMAParameters(const std::vector<double> &x0,
		    const std::vector<double> &sigma,
		    const int &lambda=-1,
		    const std::vector<double> &lbounds=std::vector<double>(),
		    const std::vector<double> &ubounds=std::vector<double>(),
		    const uint64_t &seed=0)
    :Parameters<TGenoPheno>(x0.size(),&x0.front(),lambda,seed,TGenoPheno()),_nrestarts(9),_lazy_update(false),_lazy_value(0),_cm(1.0),_alphacov(2.0),_alphaminusold(0.5),_lambdamintarget(0.66),_alphaminusmin(1.0)
  {
    dVec scaling = dVec::Constant(x0.size(),1.0).cwiseQuotient(Eigen::Map<dVec>(const_cast<double*>(&sigma.front()),sigma.size()));
    dVec shift = dVec::Constant(x0.size(),0.0);
    TGenoPheno gp(scaling,shift,&lbounds.front(),&ubounds.front()); // XXX: is only effective when GenoPheno has linScalingStrategy
    this->set_gp(gp);
    _sigma_init = *std::min_element(sigma.begin(),sigma.end());
    initialize_parameters();
  };
      
      ~CMAParameters(){};
      
      /**
       * \brief initialize required parameters based on dim, lambda, x0 and sigma.
       */
      void initialize_parameters(){
      _mu = floor(Parameters<TGenoPheno>::_lambda / 2.0);
      _weights = dVec::Zero(_mu);
      double sum_weights = 0.0, sq_weights = 0.0;
      for (int i=0;i<_mu;i++){
          _weights[i] = log(_mu+1)-log(i+1);
	        sum_weights += _weights[i];
	        sq_weights += _weights[i] * _weights[i];
      }
      _weights /= sum_weights;

    //debug
    //std::cerr << "weights: " << _weights << std::endl;
    //debug
    
    _muw = sum_weights*sum_weights / sq_weights;

    if (Parameters<TGenoPheno>::_dim < 1000)
      _csigma = (_muw+2.0)/(Parameters<TGenoPheno>::_dim+_muw+5.0);
    else _csigma = (std::sqrt(_muw) + 2.0) / (std::sqrt(Parameters<TGenoPheno>::_dim) + std::sqrt(_muw) + 3.0);
    _cc = (4.0+_muw/static_cast<double>(Parameters<TGenoPheno>::_dim))/(Parameters<TGenoPheno>::_dim+4.0+2.0*_muw/static_cast<double>(Parameters<TGenoPheno>::_dim));
    
    _c1 = 2.0/(pow((Parameters<TGenoPheno>::_dim+1.3),2)+_muw);
    _cmu = 2.0*(_muw-2.0+1.0/_muw)/(pow(Parameters<TGenoPheno>::_dim+2.0,2)+_muw);
    _cmu = std::min(1.0-_c1,_cmu);
    
    _dsigma = 1.0+_csigma+2.0*std::max(0.0,sqrt((_muw-1)/(Parameters<TGenoPheno>::_dim+1))-1);
    
    // constants used in covariance update.
    _fact_ps = sqrt(_csigma*(2.0-_csigma)*_muw);
    _fact_pc = sqrt(_cc * (2.0 - _cc) * _muw);
    
    _chi = sqrt(static_cast<double>(Parameters<TGenoPheno>::_dim))*(1.0-1.0/(4.0*Parameters<TGenoPheno>::_dim) + 1.0/(21.0*Parameters<TGenoPheno>::_dim*Parameters<TGenoPheno>::_dim));
    
    _lazy_value = 1.0/(_c1+_cmu)/Parameters<TGenoPheno>::_dim/10.0;

    // active cma.
    _deltamaxsigma = std::numeric_limits<double>::max();

    // uncertainty handling.
    this->_rlambda = std::max(0.1,2.0/Parameters<TGenoPheno>::_lambda);
    this->_alphathuh = 1 + 2.0/(Parameters<TGenoPheno>::_dim+10.0);
  };

      
      void reset_as_fixed(const int &k)
  {
    Parameters<TGenoPheno>::_dim--;
    removeElement(Parameters<TGenoPheno>::_x0min,k); // XXX: could go into parameters.cc
    removeElement(Parameters<TGenoPheno>::_x0max,k);
    removeElement(_weights,k);
  };
      
      /**
       * \brief adapt parameters for noisy objective function.
       */
      void set_noisy(){
    static double factor = 0.2;
    static double lfactor = 5.0; // lambda factor.
    Parameters<TGenoPheno>::_lambda *= lfactor;
    initialize_parameters(); // reinit parameters.
    _c1 *= factor;
    _cmu = std::min(1.0-_c1,2.0*factor*(_muw-2.0+1.0/_muw)/(pow(Parameters<TGenoPheno>::_dim+2.0,2)+_muw));
  };
      
      /**
       * \brief sets the optimization algorithm.
       *        Note: overrides Parameters::set_algo
       * @param algo from CMAES_DEFAULT, IPOP_CMAES, BIPOP_CMAES, aCMAES, aIPOP_CMAES, aBIPOP_CMAES, sepCMAES, sepIPOP_CMAES, sepBIPOP_CMAES, sepaCMAES, sepaIPOP_CMAES, sepaBIPOP_CMAES, VD_CMAES, VD_IPOP_CMAES, VD_BIPOP_CMAES 
       */
      void set_algo(const int &algo)
      {
	this->_algo = algo;
	/*if (this->_tpa != 0
	    && (this->_algo == 6 // sepCMAES
		|| this->_algo == 7 //sepIPOP_CMAES
		|| this->_algo == 8 //sepBIPOP_CMAES
		|| this->_algo == 9 //sepaCMAES
		|| this->_algo == 10 //sepaIPOP_CMAES
		|| this->_algo == 11 //sepBIPOP_CMAES
		|| this->_algo == 12 //VD_CMAES
		|| this->_algo == 13 //VD_IPOP_CMAES
		|| this->_algo == 14)) //VD_BIPOP_CMAES
		set_tpa(2); */ // XXX: deactivated until flaw is fixed
      }

      /**
       * \brief sets the optimization algorithm.
       * @param algo as string from cmaes,ipop,bipop,acmaes,aipop,abipop,sepcmaes,sepipop,sepbipop,sepacmaes,sepaipop,sepabipop,vdcma,vdipopcma,vdbipopcma
       */
      void set_str_algo(const std::string &algo)
      {
	std::map<std::string,int>::const_iterator mit;
	if ((mit = Parameters<TGenoPheno>::_algos.find(algo))!=Parameters<TGenoPheno>::_algos.end())
	  Parameters<TGenoPheno>::_algo = (*mit).second;
	else LOG(ERROR) << "unknown algorithm " << algo << std::endl;
	if (algo.find("sep")!=std::string::npos)
	  set_sep();
	if (algo.find("vd")!=std::string::npos)
	  set_vd();
      }

      /**
       * \brief activates the gradient injection scheme. 
       *        If no gradient function is defined, injects a numerical gradient solution instead
       *        Note: overrides Parameters::set_gradient
       * @param gradient true/false
       */
      void set_gradient(const bool &gradient)
      {
	this->_with_gradient = gradient;
	/*if (this->_tpa != 0)
	  set_tpa(2);*/ // TPA default when gradient is activated.
      }
      
      /**
       * \brief fix parameters for sep-CMA-ES, using only the diagonal of covariance matrix.
       */
      void set_sep() {
    _sep = true;
    _c1 *= (Parameters<TGenoPheno>::_dim+2.0)/3.0;
    _cmu = std::min(1.0-_c1,(0.3 + _muw - 2.0 + 1.0/_muw) / (this->_dim + 4.0*std::sqrt(this->_dim) + _muw/2.0));
    _lazy_value = 1.0/(_c1+_cmu)/Parameters<TGenoPheno>::_dim/10.0;
    _cc = (1.0+1.0/static_cast<double>(Parameters<TGenoPheno>::_dim) + _muw/static_cast<double>(Parameters<TGenoPheno>::_dim))/(std::sqrt(Parameters<TGenoPheno>::_dim)+1.0/static_cast<double>(Parameters<TGenoPheno>::_dim)+2.0*_muw/static_cast<double>(Parameters<TGenoPheno>::_dim));
  };

      /**
       * \brief whether algorithm leverages separability.
       * @return separability status
       */
      bool is_sep() const { return _sep; }

      /**
       * \brief activates VD decomposition.
       */
      void set_vd(){
    if (this->_algo != 12 && this->_algo != 13 && this->_algo != 14)
      {
	std::cerr << "[Warning]: set_vd on non VD algorithm " << this->_algo << ". Not activating VD update\n";
	return;
      }
    _vd = true;
    _csigma = std::sqrt(_muw)/(2.0*(std::sqrt(Parameters<TGenoPheno>::_dim) + std::sqrt(_muw)));
    _c1 *= (Parameters<TGenoPheno>::_dim-5)/6.0;
    _cmu = std::min(1.0-_c1,(Parameters<TGenoPheno>::_dim-5)/6.0*(2.0*(_muw-2.0+1.0/_muw)/(pow(Parameters<TGenoPheno>::_dim+2.0,2)+_muw)));
    if (this->_tpa == 0)
      _dsigma = 1.0+_csigma+2.0*std::max(0.0,sqrt((_muw-1)/(Parameters<TGenoPheno>::_dim+1))-1);
    _fact_ps = sqrt(_csigma*(2.0-_csigma)*_muw);
  };

      /**
       * \brief whether algorithm uses vd update.
       * @return vd status
       */
      bool is_vd() const { return _vd; }
      
      /**
       * \brief freezes a parameter to a given value during optimization.
       *        Adapts some generic parameters as well.
       * @param index dimension index of the parameter to be frozen
       * @param value frozen value of the parameter
       */
      void set_fixed_p(const int &index, const double &value) {
    Parameters<TGenoPheno>::set_fixed_p(index,value);
    double ndim = Parameters<TGenoPheno>::_dim-Parameters<TGenoPheno>::_fixed_p.size();
    _chi = sqrt(ndim)*(1.0-1.0/(4.0*ndim) + 1.0/(21.0*ndim*ndim));
    _lazy_value = 1.0/(_c1+_cmu)/ndim/10.0;
  };
      
      /**
       * \brief unfreezes a parameter.
       * @param index dimenion index of the parameter to unfreeze
       */
      void unset_fixed_p(const int &index) {
    Parameters<TGenoPheno>::unset_fixed_p(index);
    double ndim = Parameters<TGenoPheno>::_dim-Parameters<TGenoPheno>::_fixed_p.size();
    _chi = sqrt(ndim)*(1.0-1.0/(4.0*ndim) + 1.0/(21.0*ndim*ndim));
    _lazy_value = 1.0/(_c1+_cmu)/ndim/10.0;
  };
      
      /**
       * \brief sets the maximum number of restarts (applies to IPOP and BIPOP).
       * @param nrestarts maximum number of restarts
       */
      inline void set_restarts(const int &nrestarts) { _nrestarts = nrestarts; }

      /**
       * \brief get the number of restarts (applies to IPOP and BIPOP).
       * @return number of restarts
       */
      inline int get_restarts() const { return _nrestarts; }

      /**
       * \brief sets the lazy update (i.e. updates the eigenvalues every few steps).
       * @param lz whether to activate the lazy update
       */
      inline void set_lazy_update(const bool &lz) { _lazy_update = lz; }

      /**
       * \brief get lazy update status.
       * @param whether lazy update is activated
       */
      inline bool get_lazy_update() { return _lazy_update; }

      /**
       * \brief sets elitism:
       *        0 -> no elitism
       *        1 -> elitism: reinjects the best-ever seen solution
       *        2 -> initial elitism: reinject x0 as long as it is not improved upon
       *        3 -> initial elitism on restart: restart if best encountered solution is not the
       *             the final solution and reinjects the best solution until the population
       *             has better fitness, in its majority
       */
      inline void set_elitism(const int &e)
      {
	if (e == 0)
	  _elitist = _initial_elitist = _initial_elitist_on_restart;
	else if (e == 1)
	  {
	    _elitist = true;
	    _initial_elitist = _initial_elitist_on_restart = false;
	  }
	else if (e == 2)
	  {
	    _initial_elitist = true;
	    _elitist = _initial_elitist_on_restart = false;
	  }
	else if (e == 3)
	  {
	    _initial_elitist_on_restart = true;
	    _elitist = _initial_elitist = false;
	  }
      }
      
      /**
       * \brief all stopping criteria are active by default, this allows to control
       *        them
       * @param criteria stopping criteria CMAStopCritType, see cmastopcriteria.h
       * @param active whether to activate this criteria
       */
      inline void set_stopping_criteria(const int &criteria,
					const bool &active)
      {
	_stoppingcrit.insert(std::pair<int,bool>(criteria,active));
      }

      /**
       * \brief activates / deactivates two-point adaptation step-size mechanism.
       *        Overrides parameters::set_tpa by automatically setting dsigma value.
       * @param b 0: no, 1: auto, 2: yes
       */
      void set_tpa(const int &b){
    this->_tpa = b;
    if (this->_tpa == 2)
      _dsigma = 4 - 3.6 / std::sqrt(Parameters<TGenoPheno>::_dim);
    else if (!this->_vd)
      _dsigma = 1.0+_csigma+2.0*std::max(0.0,sqrt((_muw-1)/(Parameters<TGenoPheno>::_dim+1))-1); // default
    else _dsigma = 1.0+_csigma+2.0*std::max(0.0,sqrt((_muw-1)/(Parameters<TGenoPheno>::_dim+1))-1); // vd
  }; // overrides def in parameters.h in order to reset dsigma

      /**
       * \brief sets dsigma value, use with care.
       * @param d dsigma
       */
      void set_tpa_dsigma(const double &d) { _dsigma = d; }
      
    public:
      int _mu; /**< number of candidate solutions used to update the distribution parameters. */
      dVec _weights; /**< offsprings weighting scheme. */
      double _csigma; /**< cumulation constant for step size. */
      double _c1; /**< covariance matrix learning rate for the rank one update using pc. */
      double _cmu; /**< covariance matrix learning reate for the rank mu update. */
      double _cc; /**< cumulation constant for pc. */
      double _muw; /**< \sum^\mu _weights .*/
      double _dsigma; /**< step size damping factor. */
      
      // computed once at init for speeding up operations.
      double _fact_ps;
      double _fact_pc;
      double _chi; /**< norm of N(0,I) */
      
      double _sigma_init; /**< initial sigma value. */
      
      int _nrestarts = 9; /**< maximum number of restart, when applicable. */
      bool _lazy_update; /**< covariance lazy update. */
      double _lazy_value; /**< reference trigger for lazy update. */
      
      // active cma.
      double _cm; /**< learning rate for the mean. */
      double _alphacov; /**< = 2 (active CMA only) */
      double _alphaminusold; /**< in [0,1] (active CMA only) */
      double _deltamaxsigma; /**< infinite (active CMA only) */
      double _lambdamintarget; /**< = 0.66 (active CMA only) */
      double _alphaminusmin; /**< = 1 (active CMA only) */
      
      // sep cma (diagonal cov).
      bool _sep = false; /**< whether to use diagonal covariance matrix. */
      bool _vd = false;
      
      bool _elitist = false; /**< re-inject the best-ever seen solution. */
      bool _initial_elitist = false; /**< re-inject x0. */
      bool _initial_elitist_on_restart = false; /**< activate the restart from and re-injection of the best seen solution if not the final one. */
      
      // stopping criteria
      std::map<int,bool> _stoppingcrit; /**< control list of stopping criteria. */

      // kl
      bool _kl = false;
    };

  template<class TGenoPheno>
    std::map<std::string,int> Parameters<TGenoPheno>::_algos = {{"cmaes",0},{"ipop",1},{"bipop",2},{"acmaes",3},{"aipop",4},{"abipop",5},{"sepcmaes",6},{"sepipop",7},{"sepbipop",8},{"sepacmaes",9},{"sepipop",10},{"sepbipop",11},{"vdcma",12},{"vdipopcma",13},{"vdbipopcma",14}};


}
#endif
