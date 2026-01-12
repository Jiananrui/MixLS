// src/mels_loglik.cpp

#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <limits>

// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]

using namespace arma;
using namespace RcppParallel;

// -----------------------------------------------------------------------------
// WORKER 1: Stage 1 & 2 Log-Likelihood (Both Fixed and Adaptive)
// -----------------------------------------------------------------------------

struct Stage12LoglikWorker : public Worker {
    // Inputs
    const vec& params;
    const mat& X;
    const mat& U;
    const mat& Z;
    const vec& y;
    const uvec& id;
    const uvec& unique_ids;
    const vec& points;
    const vec& weights;
    const int adaptive;
    const vec& eb_theta; // Only used if adaptive=1
    const vec& eb_sd;   // Only used if adaptive=1

    // Output
    double value;

    // Internal
    int n_fixed, n_random, nq;

    // Constructor
    Stage12LoglikWorker(const vec& params, const mat& X, const mat& U, const mat& Z,
                        const vec& y, const uvec& id, const uvec& unique_ids,
                        const vec& points, const vec& weights, int adaptive,
                        const vec& eb_theta, const vec& eb_sd)
        : params(params), X(X), U(U), Z(Z), y(y), id(id), unique_ids(unique_ids),
          points(points), weights(weights), adaptive(adaptive),
          eb_theta(eb_theta), eb_sd(eb_sd), value(0.0) 
    {
        n_fixed = X.n_cols;
        n_random = U.n_cols;
        nq = points.n_elem;
    }

    // Split constructor
    Stage12LoglikWorker(const Stage12LoglikWorker& w, Split)
        : params(w.params), X(w.X), U(w.U), Z(w.Z), y(w.y), id(w.id),
          unique_ids(w.unique_ids), points(w.points), weights(w.weights),
          adaptive(w.adaptive), eb_theta(w.eb_theta), eb_sd(w.eb_sd), 
          value(0.0)
    {
        n_fixed = w.n_fixed;
        n_random = w.n_random;
        nq = w.nq;
    }

    // The parallel loop
    void operator()(std::size_t begin, std::size_t end) {
        
        vec beta = params.subvec(0, n_fixed - 1);
        vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
        vec tau = params.subvec(n_fixed + n_random, params.n_elem - 1);

        double local_loglik = 0.0;

        for (std::size_t i = begin; i < end; i++) {
            uvec idx = find(id == unique_ids(i));
            
            mat X_i = X.rows(idx);
            mat U_i = U.rows(idx);
            mat Z_i = Z.rows(idx);
            vec y_i = y.elem(idx);
            
            vec U_alpha_i = U_i * alpha;
            vec Z_tau_i = Z_i * tau;
            vec X_beta_i = X_i * beta;
            vec sigma_v_sq_i = exp(U_alpha_i);
            vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
            vec sigma_epsilon_sq_i = exp(Z_tau_i);

            // Adaptive setup
            vec current_points = points;
            vec current_weights = weights;
            
            if (adaptive == 1 && i < eb_theta.n_elem && i < eb_sd.n_elem && eb_sd(i) > 1e-8) {
                current_points = eb_theta(i) + eb_sd(i) * points;
                current_weights = eb_sd(i) * exp(0.5 * (points % points - current_points % current_points)) % weights;
            }

            double h_i = 0.0;
            
            for (int q = 0; q < nq; q++) {
                double theta_1i = current_points(q);
                double w = current_weights(q);
                
                vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
                vec epsilon = y_i - mu;
                
                vec f_i = log(2.0 * M_PI * sigma_epsilon_sq_i) + square(epsilon) / sigma_epsilon_sq_i;
                double l_i = exp(-0.5 * sum(f_i));
                h_i += l_i * w;
            }
            
            // Safe log accumulation
            if (h_i > 0) {
                local_loglik += log(h_i);
            }
        }
        
        value += local_loglik;
    }

    // Combine sums from different threads
    void join(const Stage12LoglikWorker& rhs) {
        value += rhs.value;
    }
};

//' Calculate log-likelihood for Stage 1 & 2 (Fixed Quadrature)
//'
//' Computes the marginal log-likelihood for the mixed effects location-scale model
//' (Stage 1 & 2) using standard Gaussian quadrature.
//'
//' @param params A vector of model parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//'
//' @return The total log-likelihood (scalar double).
//' @keywords internal
// [[Rcpp::export]]
double stage12_loglik_rcpp(const arma::vec& params,
                         const arma::mat& X,
                         const arma::mat& U,
                         const arma::mat& Z,
                         const arma::vec& y,
                         const arma::uvec& id,
                         const arma::vec& points,
                         const arma::vec& weights) {
    uvec unique_ids = unique(id);
    vec dummy_eb; // Placeholder
    
    Stage12LoglikWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, 0, dummy_eb, dummy_eb);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.value;
}

//' Calculate log-likelihood for Stage 1 & 2 (Adaptive Quadrature)
//'
//' Computes the marginal log-likelihood for the mixed effects location-scale model
//' (Stage 1 & 2) using adaptive Gaussian quadrature.
//'
//' @param params A vector of model parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//' @param adaptive Integer flag (0 or 1) indicating if adaptive quadrature is used.
//' @param eb_theta Vector of Empirical Bayes estimates for random effect means.
//' @param eb_sd Vector of Empirical Bayes estimates for random effect standard deviations.
//'
//' @return The total log-likelihood (scalar double).
//' @keywords internal
// [[Rcpp::export]]
double stage12_loglik_adaptive(const arma::vec& params,
                             const arma::mat& X,
                             const arma::mat& U,
                             const arma::mat& Z,
                             const arma::vec& y,
                             const arma::uvec& id,
                             const arma::vec& points,
                             const arma::vec& weights,
                             int adaptive,
                             const arma::vec& eb_theta,
                             const arma::vec& eb_sd) {
    uvec unique_ids = unique(id);
    
    Stage12LoglikWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, adaptive, eb_theta, eb_sd);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.value;
}


// -----------------------------------------------------------------------------
// WORKER 2: Stage 3 Log-Likelihood (Both Fixed and Adaptive)
// -----------------------------------------------------------------------------

struct Stage3LoglikWorker : public Worker {
    // Inputs
    const vec& params;
    const mat& X;
    const mat& U;
    const mat& Z;
    const vec& y;
    const uvec& id;
    const uvec& unique_ids;
    const vec& points;
    const vec& weights;
    const std::string& model_type;
    const int adaptive;
    const mat& eb_theta2;
    const mat& eb_thetav;

    // Output
    double value;

    // Internal
    int n_fixed, n_random, n_ws, nq;

    // Constructor
    Stage3LoglikWorker(const vec& params, const mat& X, const mat& U, const mat& Z,
                       const vec& y, const uvec& id, const uvec& unique_ids,
                       const vec& points, const vec& weights, const std::string& model_type,
                       int adaptive, const mat& eb_theta2, const mat& eb_thetav)
        : params(params), X(X), U(U), Z(Z), y(y), id(id), unique_ids(unique_ids),
          points(points), weights(weights), model_type(model_type),
          adaptive(adaptive), eb_theta2(eb_theta2), eb_thetav(eb_thetav), value(0.0)
    {
        n_fixed = X.n_cols;
        n_random = U.n_cols;
        n_ws = Z.n_cols;
        nq = points.n_elem;
    }

    // Split constructor
    Stage3LoglikWorker(const Stage3LoglikWorker& w, Split)
        : params(w.params), X(w.X), U(w.U), Z(w.Z), y(w.y), id(w.id),
          unique_ids(w.unique_ids), points(w.points), weights(w.weights),
          model_type(w.model_type), adaptive(w.adaptive),
          eb_theta2(w.eb_theta2), eb_thetav(w.eb_thetav), value(0.0)
    {
        n_fixed = w.n_fixed;
        n_random = w.n_random;
        n_ws = w.n_ws;
        nq = w.nq;
    }

    void operator()(std::size_t begin, std::size_t end) {
        
        vec beta = params.subvec(0, n_fixed - 1);
        vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
        vec tau = params.subvec(n_fixed + n_random, n_fixed + n_random + n_ws - 1);
        vec sigma = params.subvec(n_fixed + n_random + n_ws, params.n_elem - 1);
        const double log_2pi = std::log(2.0 * M_PI);

        // Pre-allocate thread-local temp vector
        vec theta;
        if (model_type == "independent") { theta = vec(1); }
        else if (model_type == "linear") { theta = vec(2); }
        else { theta = vec(3); }

        double local_loglik = 0.0;

        for (std::size_t i = begin; i < end; i++) {
            uvec idx = find(id == unique_ids(i));
            
            vec y_i = y.elem(idx);
            mat X_i = X.rows(idx);
            mat U_i = U.rows(idx);
            mat Z_i = Z.rows(idx);
            
            vec X_beta_i = X_i * beta;
            vec Z_tau_i = Z_i * tau;
            vec U_alpha_i = U_i * alpha;
            vec sigma_v_sq_i = exp(U_alpha_i);
            vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);

            // Adaptive setup
            vec current_points_1 = points, current_weights_1 = weights;
            vec current_points_2 = points, current_weights_2 = weights;

            if (adaptive == 1 && i < eb_theta2.n_rows) {
                double mean1 = eb_theta2(i, 0);
                double mean2 = eb_theta2(i, 1);
                double var1 = eb_thetav(i, 0);
                double var2 = eb_thetav(i, 2);

                if (var1 > 1e-8) {
                    double sd1 = std::sqrt(var1);
                    current_points_1 = mean1 + sd1 * points;
                    current_weights_1 = sd1 * exp(0.5 * (points % points - square(current_points_1))) % weights;
                }
                if (var2 > 1e-8) {
                    double sd2 = std::sqrt(var2);
                    current_points_2 = mean2 + sd2 * points;
                    current_weights_2 = sd2 * exp(0.5 * (points % points - square(current_points_2))) % weights;
                }
            }

            double h_i = 0.0;

            for (int q1 = 0; q1 < nq; q1++) {
                double theta_1i = current_points_1(q1);
                double w1 = current_weights_1(q1);
                
                vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
                vec epsilon = y_i - mu;
                
                for (int q2 = 0; q2 < nq; q2++) {
                    double theta_2i = current_points_2(q2);
                    double w = w1 * current_weights_2(q2);
                    
                    if (model_type == "independent") { theta(0) = theta_2i; }
                    else if (model_type == "linear") { theta(0) = theta_1i; theta(1) = theta_2i; }
                    else if (model_type == "interaction") { theta(0) = theta_1i; theta(1) = theta_1i * theta_2i; theta(2) = theta_2i; }
                    else { theta(0) = theta_1i; theta(1) = theta_1i * theta_1i; theta(2) = theta_2i; }
                    
                    vec sigma_epsilon_sq = exp(Z_tau_i + dot(sigma, theta));
                    vec f_i = log_2pi + log(sigma_epsilon_sq) + square(epsilon) / sigma_epsilon_sq;
                    double l_i = exp(-0.5 * sum(f_i));
                    h_i += l_i * w;
                }
            }

            if (h_i > 0) {
                local_loglik += log(h_i);
            }
        }
        
        value += local_loglik;
    }

    void join(const Stage3LoglikWorker& rhs) {
        value += rhs.value;
    }
};

//' Calculate log-likelihood for Stage 3 (Fixed Quadrature)
//'
//' Computes the marginal log-likelihood for the Stage 3 model (bivariate random effects)
//' using standard Gaussian quadrature.
//'
//' @param params A vector of model parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//' @param model_type String specifying the model type ("independent", "linear", "quadratic" and "interaction").
//'
//' @return The total log-likelihood (scalar double).
//' @keywords internal
// [[Rcpp::export]]
double stage3_loglik_combined(const arma::vec& params,
                            const arma::mat& X,
                            const arma::mat& U,
                            const arma::mat& Z,
                            const arma::vec& y,
                            const arma::uvec& id,
                            const arma::vec& points,
                            const arma::vec& weights,
                            const std::string& model_type = "linear") {
    uvec unique_ids = unique(id);
    mat dummy_eb_mat;
    
    Stage3LoglikWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, model_type, 0, dummy_eb_mat, dummy_eb_mat);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.value;
}

//' Calculate log-likelihood for Stage 3 (Adaptive Quadrature)
//'
//' Computes the marginal log-likelihood for the Stage 3 model (bivariate random effects)
//' using adaptive Gaussian quadrature.
//'
//' @param params A vector of model parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//' @param model_type String specifying the model type ("independent", "linear", "quadratic" and "interaction").
//' @param adaptive Integer flag (0 or 1) indicating if adaptive quadrature is used.
//' @param eb_theta2 Matrix (n x 2) of Empirical Bayes estimates for random effect means.
//' @param eb_thetav Matrix (n x 3) of Empirical Bayes estimates for random effect variances/covariances.
//'
//' @return The total log-likelihood (scalar double).
//' @keywords internal
// [[Rcpp::export]]
double stage3_loglik_adaptive_combined(const arma::vec& params,
                                      const arma::mat& X,
                                      const arma::mat& U,
                                      const arma::mat& Z,
                                      const arma::vec& y,
                                      const arma::uvec& id,
                                      const arma::vec& points,
                                      const arma::vec& weights,
                                      const std::string& model_type,
                                      int adaptive,
                                      const arma::mat& eb_theta2,
                                      const arma::mat& eb_thetav) {
    uvec unique_ids = unique(id);
    
    Stage3LoglikWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, model_type, adaptive, eb_theta2, eb_thetav);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.value;
}