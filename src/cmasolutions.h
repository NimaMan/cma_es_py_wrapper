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

#ifndef CMASOLUTIONS_H
#define CMASOLUTIONS_H

#include "libcmaes_config.h"
#include "candidate.h"
#include "eo_matrix.h"
#include "cmaparameters.h"
#include "pli.h"
#include "opti_err.h"
#include "eigenmvn.h"

#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>


namespace libcmaes
{
  
  /**
   * \brief Holder of the set of evolving solutions from running an instance of CMA-ES.
   */
  class CMASolutions
  {
    friend class CMAES;
    template <class U, class V> friend class CMAStrategy;
    template <class U, class V> friend class ESOptimizer;
    template <class U, class V> friend class ESOStrategy;
    template <class U, class V> friend class IPOPCMAStrategy;
    template <class U, class V> friend class BIPOPCMAStrategy;
    friend class CovarianceUpdate;
    friend class ACovarianceUpdate;
    template <class U> friend class errstats;
#ifdef HAVE_SURROG
    template <template <class X,class Y> class U, class V, class W> friend class SimpleSurrogateStrategy;
    template <template <class X,class Y> class U, class V, class W> friend class ACMSurrogateStrategy;
#endif
    friend class VDCMAUpdate;
    
  public:
    /**
     * \brief dummy constructor.
     */
    CMASolutions() {}

    /**
     * \brief initializes solutions from stochastic optimization parameters.
     * @param p parameters
     * TODO: the template parameters are different from that of the .cc file
     */
    template<class TGenoPheno=GenoPheno<NoBoundStrategy>>
    CMASolutions(Parameters<TGenoPheno> &p) 
    :_hsig(1),_max_eigenv(0.0),_min_eigenv(0.0),_niter(0),_nevals(0),_kcand(1),_eigeniter(0),_updated_eigen(true),_run_status(0),_elapsed_time(0)
    {
      try{
          	if (!static_cast<CMAParameters<TGenoPheno>&>(p)._sep && !static_cast<CMAParameters<TGenoPheno>&>(p)._vd)
	              _cov = dMat::Identity(p._dim,p._dim);
	          else _sepcov = dMat::Constant(p._dim,1,1.0);
      }
      catch (std::bad_alloc &e){
	        _run_status = OPTI_ERR_OUTOFMEMORY;
	        return;
      }
      if (p._x0min == p._x0max){
        	if (p._x0min == dVec::Constant(p._dim,-std::numeric_limits<double>::max()))
	        _xmean = dVec::Random(p._dim) * 4.0; // initial mean randomly sampled from -4,4 in all dimensions.
	        else _xmean = p._x0min;
          }
      else{
        	_xmean = 0.5*(dVec::Random(p._dim) + dVec::Constant(p._dim,1.0)); // scale to [0,1].
          _xmean = _xmean.cwiseProduct(p._x0max - p._x0min) + p._x0min; // scale to bounds.
      }
      if (!p._fixed_p.empty()){
          auto fpmit = p._fixed_p.begin();
          while (fpmit!=p._fixed_p.end()){
              _xmean((*fpmit).first) = (*fpmit).second;
              ++fpmit;
          }
      }
      // if scaling, need to apply to xmean.
      if (!p._gp._scalingstrategy._id)
        p._gp._scalingstrategy.scale_to_internal(_xmean,_xmean);
      if (static_cast<CMAParameters<TGenoPheno>&>(p)._sigma_init > 0.0)
        _sigma = static_cast<CMAParameters<TGenoPheno>&>(p)._sigma_init;
      else static_cast<CMAParameters<TGenoPheno>&>(p)._sigma_init = _sigma = 1.0/static_cast<double>(p._dim); // XXX: sqrt(trace(cov)/dim)
      
      _psigma = dVec::Zero(p._dim);
      _pc = dVec::Zero(p._dim);
      _candidates.resize(p._lambda);
      _kcand = std::min(p._lambda-1,static_cast<int>(1.0+ceil(0.1+p._lambda/4.0)));
      _max_hist = (p._max_hist > 0) ? p._max_hist : static_cast<int>(10+ceil(30*p._dim/p._lambda));
      
      if (static_cast<CMAParameters<TGenoPheno>&>(p)._vd){
          Eigen::EigenMultivariateNormal<double> esolver(false,static_cast<uint64_t>(p._seed));
          esolver.set_covar(_sepcov);
          _v = esolver.samples_ind(1) / std::sqrt(p._dim);
      }
  }

    
    ~CMASolutions(){};

    /**
     * \brief sorts the current internal set of solution candidates.
     */
    void sort_candidates()
    {
      std::stable_sort(_candidates.begin(),_candidates.end(),
		[](Candidate const &c1, Candidate const &c2){return c1.get_fvalue() < c2.get_fvalue();});
    }

    /**
     * \brief updates the history of best candidates, as well as other meaningful
     *        values, typically used in termination criteria.
     * @see CMAStopCriteria
     */
    void update_best_candidates(){
          _best_candidates_hist.push_back(_candidates.at(0)); // supposed candidates is sorted.
          _k_best_candidates_hist.push_back(_candidates.at(_kcand));
          if ((int)_best_candidates_hist.size() > _max_hist)
            {
        _best_candidates_hist.erase(_best_candidates_hist.begin());
        _k_best_candidates_hist.erase(_k_best_candidates_hist.begin());
            }
          
          _bfvalues.push_back(_candidates.at(0).get_fvalue());
          if (_bfvalues.size() > 20)
            _bfvalues.erase(_bfvalues.begin());

          // get median of candidate's scores, used in termination criteria (stagnation).
          double median = 0.0;
          size_t csize = _candidates.size();
          if (csize % 2 == 0)
            median = (_candidates[csize/2-1].get_fvalue() + _candidates[csize/2].get_fvalue())/2.0;
          else median = _candidates[csize/2].get_fvalue();
          _median_fvalues.push_back(median);
          if (_median_fvalues.size() > static_cast<size_t>(ceil(0.2*_niter+120+30*_xmean.size()/static_cast<double>(_candidates.size()))))
            _median_fvalues.erase(_median_fvalues.begin());

          // store best seen candidate.
          if ((_niter == 0 && !_best_seen_candidate.get_x_size()) || _candidates.at(0).get_fvalue() < _best_seen_candidate.get_fvalue())
            {
        _best_seen_candidate = _candidates.at(0);
        _best_seen_iter = _niter;
            }

          // store the worst seen candidate.
          if ((_niter == 0 && !_worst_seen_candidate.get_x_size()) || _candidates.back().get_fvalue() > _worst_seen_candidate.get_fvalue())
            {
        _worst_seen_candidate = _candidates.back();
            }
    };

    /**
     * \brief updates reference eigenvalue and eigenvectors, for use in 
     *        termination criteria.
     * @see CMAStopCriteria
     */
    void update_eigenv(const dVec &eigenvalues,
		       const dMat &eigenvectors){
    _max_eigenv = eigenvalues.maxCoeff();
    _min_eigenv = eigenvalues.minCoeff();
    _leigenvalues = eigenvalues;
    _leigenvectors = eigenvectors;
  }

    /**
     * \brief returns current best solution candidate.
     *        NOTE: candidates MUST be sorted
     * @return current best candidate
     * @see CMASolutions::sort_candidates
     */
    inline Candidate best_candidate() const
    {
      if (_best_candidates_hist.empty()) // iter = 0
	{
	  if (_initial_candidate.get_x_size())
	    return _initial_candidate;
	  else return Candidate(std::numeric_limits<double>::quiet_NaN(),_xmean);
	}
      return _best_candidates_hist.back();
    }

    /**
     * \brief returns the best seen candidate.
     * @return best seen candidate
     */
    inline Candidate get_best_seen_candidate() const
    {
      return _best_seen_candidate;
    }

    /**
     * \brief returns the worst seen candidate.
     * @return worst seen candidate
     */
    inline Candidate get_worst_seen_candidate() const
    {
      return _worst_seen_candidate;
    }

    /**
     * \brief get a reference to the r-th candidate in current set
     * @param r candidate position
     */
    inline Candidate& get_candidate(const int &r)
      {
	return _candidates.at(r);
      }

    inline Candidate get_candidate(const int &r) const
    {
      return _candidates.at(r);
    }

    /**
     * \brief get a reference to the full candidate set
     */
    inline std::vector<Candidate>& candidates()
    {
      return _candidates;
    }
    
    /**
     * \brief number of candidate solutions.
     * @return current number of solution candidates.
     */
    inline int size() const
    {
      return _candidates.size();
    }

    /**
     * \brief resets the solution object in order to restart from
     *        the current solution with fresh covariance matrix.
     * Note: experimental.
     */
    void reset(){
        //_candidates.clear();
        _best_candidates_hist.clear();
        //_leigenvalues.setZero(); // beware.
        //_leigenvectors.setZero();
        //_cov /= 1e-3;//_sigma;
            if (_csqinv.rows())
            _cov = dMat::Identity(_csqinv.rows(),_csqinv.cols());
        else _sepcov = dMat::Constant(_sepcsqinv.rows(),1,1.0);
        //std::cout << "cov: " << _cov << std::endl;
        _niter = 0;
        _nevals = 0;
        //_sigma = 1.0/static_cast<double>(_csqinv.rows());
        if (_cov.rows())
          {
      _psigma = dVec::Zero(_cov.rows());
      _pc = dVec::Zero(_cov.rows());
          }
        else
          {
      _psigma = dVec::Zero(_sepcov.rows());
      _pc = dVec::Zero(_sepcov.rows());
          }
        _k_best_candidates_hist.clear();
        _bfvalues.clear();
        _median_fvalues.clear();
        _run_status = 0;
        _elapsed_time = _elapsed_last_iter = 0;
      #ifdef HAVE_DEBUG
          _elapsed_eval = _elapsed_ask = _elapsed_tell = _elapsed_stop = 0;
      #endif
    };
      
    /**
     * \brief re-arrange solution object such that parameter 'k' is fixed (i.e. removed).
     * @param k index of the parameter to remove.
     */
    void reset_as_fixed(const int &k){
    removeRow(_cov,k);
    removeColumn(_cov,k);
    removeRow(_csqinv,k);
    removeColumn(_csqinv,k);
    removeElement(_xmean,k);
    removeElement(_psigma,k);
    removeElement(_pc,k);
    for (size_t i=0;i<_candidates.size();i++)
      removeElement(_candidates.at(i).get_x_dvec_ref(),k);
    _best_candidates_hist.clear();
    removeElement(_leigenvalues,k);
    removeRow(_leigenvectors,k);
    removeColumn(_leigenvectors,k);
    _niter = 0;
    _nevals = 0;
    _k_best_candidates_hist.clear();
    _bfvalues.clear();
    _median_fvalues.clear();
    _run_status = 0;
    _elapsed_time = _elapsed_last_iter = 0;
#ifdef HAVE_DEBUG
    _elapsed_eval = _elapsed_ask = _elapsed_tell = _elapsed_stop = 0;
#endif
  };

    /**
     * \brief get profile likelihood if previously computed.
     */
    bool get_pli(const int &k, pli &p) const
    {
      std::map<int,pli>::const_iterator mit;
      if ((mit=_pls.find(k))!=_pls.end())
	{
	  p = (*mit).second;
	  return true;
	}
      return false;
    }

    /**
     * \brief return problem dimension.
     * @return problem dimension
     */
    inline int dim() const
    {
      return _xmean.size();
    }
    
    /**
     * \brief returns expected distance to minimum.
     * @return edm
     */
    inline double edm() const
    {
      return _edm;
    }

    /**
     * \brief returns error covariance matrix
     * @return error covariance matrix
     */
    inline dMat cov() const
    {
      return _cov;
    }

    /**
     * \brief returns reference to error covariance matrix
     * @return error covariance matrix
     */
    inline const dMat& cov_ref() const
    {
      return _cov;
    }
    
    /**
     * \brief returns pointer to covariance matrix array
     * @return pointer to covariance matrix array
     */
    inline const double* cov_data() const
    {
      return _cov.data();
    }
    
    /**
     * \brief returns separable covariance diagonal matrix, only applicable to sep-CMA-ES algorithms.
     * @return error covariance diagonal vector
     */
    inline dMat sepcov() const
    {
      return _sepcov;
    }

    /**
     * \brief returns reference to separable covariance diagonal vector, only applicable to sep-CMA-ES algorithms.
     * @return error covariance diagonal vector
     */
    inline const dMat& sepcov_ref() const
    {
      return _sepcov;
    }
    
    /**
     * \brief returns pointer to covariance diagnoal vector
     * @return pointer to covariance diagonal array
     */
    inline const double* sepcov_data() const
    {
      return _sepcov.data();
    }

    /**
     * \brief returns inverse root square of covariance matrix
     * @return square root of error covariance matrix
     */
    inline dMat csqinv() const
    {
      return _csqinv;
    }

    /**
     * \brief returns inverse root square of separable covariance diagonal matrix, only applicable to sep-CMA-ES algorithms.
     * @return square root of error covariance diagonal matrix
     */
    inline dMat sepcsqinv() const
    {
      return _sepcsqinv;
    }
    
    /**
     * \brief returns current value of step-size sigma
     * @return current step-size
     */
    inline double sigma() const
    {
      return _sigma;
    }

    /**
     * \brief set sigma value
     * For careful use, sigma is normally automatically set by strategies
     * @param s new value for sigma
     */
    inline void set_sigma(const double &s)
    {
      _sigma = s;
    }

    /**
     * \brief returns current distribution's mean in parameter space
     * @return mean
     */
    inline dVec xmean() const
    {
      return _xmean;
    }

    /**
     * \brief sets the current distributions' mean in parameter space
     * @param xmean mean vector
     */
    inline void set_xmean(const dVec &xmean)
    {
      _xmean = xmean;
    }
    
    /**
     * \brief returns current optimization status.
     * @return status
     */
    inline int run_status() const
    {
      return _run_status;
    }

    /**
     * \brief returns current optimization status' message.
     * @return status message
     */
    inline std::string status_msg() const
      {
	        return "NotImplemented";
      }

    /**
     * \brief returns currently elapsed time spent on optimization
     * @return time spent on optimization
     */
    inline int elapsed_time() const
    {
      return _elapsed_time;
    }

    /**
     * \brief returns time spent on last iteration
     * @return time spent on last iteration
     */
    inline int elapsed_last_iter() const
    {
      return _elapsed_last_iter;
    }
    
    /**
     * \brief returns current number of iterations
     * @return number of iterations
     */
    inline int niter() const
    {
      return _niter;
    }

    /**
     * \brief returns current budget (number of objective function calls)
     * @return number of objective function calls
     */
    inline int nevals() const
    {
      return _nevals;
    }
    
    /**
     * \brief returns current minimal eigen value
     * @return minimal eigen value
     */
    inline double min_eigenv() const
    {
      return _min_eigenv;
    }

    /**
     * \brief returns current maximal eigen value
     * @return maximal eigen value
     */
    inline double max_eigenv() const
    {
      return _max_eigenv;
    }

    /**
     * \brief returns whether the last update is lazy
     * @return whether the last update is lazy
     */
    inline bool updated_eigen() const
    {
      return _updated_eigen;
    }

    /**
     * \brief returns current number of objective function evaluations
     * @return number of objective function evaluations
     */
    inline int fevals() const
    {
      return _nevals;
    }

    /**
     * \brief returns last computed eigenvalues
     * @return last computed eigenvalues
     */
    inline dVec eigenvalues() const
    {
      return _leigenvalues;
    }
    
    /**
     * \brief returns last computed eigenvectors
     * @return last computed eigenvectors
     */
    inline dMat eigenvectors() const
    {
      return _leigenvectors;
    }

    /**
     * \brief print the solution object out.
     * @param out output stream
     * @param verb_level verbosity level: 0 for short, 1 for debug.
     */
    template <class TGenoPheno=GenoPheno<NoBoundStrategy>>
    std::ostream& print(std::ostream &out,
			const int &verb_level=0,
			const TGenoPheno &gp=GenoPheno<NoBoundStrategy>()) const  {
    if (_candidates.empty())
      {
	return out;
      }
    out << "best solution => f-value=" << best_candidate().get_fvalue() << " / fevals=" << _nevals << " / sigma=" << _sigma << " / iter=" << _niter << " / elaps=" << _elapsed_time << "ms" << " / x=" << gp.pheno(best_candidate().get_x_dvec()).transpose();
    if (verb_level)
      {
	out << "\ncovdiag=" << _cov.diagonal().transpose() << std::endl;
	out << "psigma=" << _psigma.transpose() << std::endl;
	out << "pc=" << _pc.transpose() << std::endl;
      }
    if (!_pls.empty())
      {
	out << "\nconfidence intervals:\n";
	for (auto it=_pls.begin();it!=_pls.end();++it)
	  {
	    out << "dim " << (*it).first << " in [" << (*it).second._min << "," << (*it).second._max << "] with error [" << (*it).second._errmin << "," << (*it).second._errmax << "]";
	    if ((*it).second._err[(*it).second._minindex] || (*it).second._err[(*it).second._maxindex])
	      out << " / status=[" << (*it).second._err[(*it).second._minindex] << "," << (*it).second._err[(*it).second._maxindex] << "]";
	    out << " / fvalue=" << "(" << (*it).second._fvaluem((*it).second._minindex) << "," << (*it).second._fvaluem((*it).second._samplesize+1+(*it).second._maxindex) << ")\n";
	    if (verb_level)
	      {
		out << "x=" << "([" << (*it).second._xm.row((*it).second._minindex) << "],[" << (*it).second._xm.row((*it).second._samplesize + 1 + (*it).second._maxindex) << "])\n";
	      }
	  }
      }
    return out;
  };

  public:
    dMat _cov; /**< covariance matrix. */
    dMat _csqinv; /** inverse root square of covariance matrix. */
    dMat _sepcov;
    dMat _sepcsqinv;
    dVec _xmean; /**< distribution mean. */
    dVec _psigma; /**< cumulation for sigma. */
    dVec _pc; /**< cumulation for covariance. */
    short _hsig = 1; /**< 0 or 1. */
    double _sigma; /**< step size. */
    std::vector<Candidate> _candidates; /**< current set of candidate solutions. */
    std::vector<Candidate> _best_candidates_hist; /**< history of best candidate solutions. */
    int _max_hist = -1; /**< max size of the history, keeps memory requirements fixed. */
    
    double _max_eigenv = 0.0; /**< max eigenvalue, for termination criteria. */
    double _min_eigenv = 0.0; /**< min eigenvalue, for termination criteria. */
    dVec _leigenvalues; /**< last computed eigenvalues, for termination criteria. */
    dMat _leigenvectors; /**< last computed eigenvectors, for termination criteria. */
    int _niter = 0; /**< number of iterations to reach this solution, for termination criteria. */
    int _nevals = 0; /**< number of function calls to reach the current solution. */
    int _kcand = 1;
    std::vector<Candidate> _k_best_candidates_hist; /**< k-th best candidate history, for termination criteria, k is kcand=1+floor(0.1+lambda/4). */
    std::vector<double> _bfvalues; /**< best function values over the past 20 steps, for termination criteria. */
    std::vector<double> _median_fvalues; /**< median function values of some steps, in the past, for termination criteria. */
    
    int _eigeniter = 0; /**< eigenvalues computation last step, lazy-update only. */
    bool _updated_eigen = true; /**< last update is not lazy. */

    // status of the run.
    int _run_status = 0; /**< current status of the stochastic optimization (e.g. running, or stopped under termination criteria). */
    int _elapsed_time = 0; /**< final elapsed time of stochastic optimization. */
    int _elapsed_last_iter = 0; /**< time consumed during last iteration. */
#ifdef HAVE_DEBUG
    int _elapsed_eval = 0;
    int _elapsed_ask = 0;
    int _elapsed_tell = 0;
    int _elapsed_stop = 0;
#endif

    std::map<int,pli> _pls; /**< profile likelihood for parameters it has been computed for. */
    double _edm = 0.0; /**< expected vertical distance to the minimum. */

    Candidate _best_seen_candidate; /**< best seen candidate along the run. */
    int _best_seen_iter;
    Candidate _worst_seen_candidate;
    Candidate _initial_candidate;
    
    dVec _v; /**< complementary vector for use in vdcma. */

    std::vector<RankedCandidate> _candidates_uh; /**< temporary set of candidates used by uncertainty handling scheme. */
    int _lambda_reev; /**< number of reevaluated solutions at current step. */
    double _suh; /**< uncertainty level computed by uncertainty handling procedure. */
    
    double _tpa_s = 0.0;
    int _tpa_p1 = 0;
    int _tpa_p2 = 1;
    dVec _tpa_x1;
    dVec _tpa_x2;
    dVec _xmean_prev; /**< previous step's mean vector. */
  };

  std::ostream& operator<<(std::ostream &out,const CMASolutions &cmas)  {
    cmas.print(out,0);
        return out;
  };
  


}

#endif
