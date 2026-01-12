// src/mels_gradient.cpp

#include <RcppArmadillo.h>
#include <RcppParallel.h>

// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]

using namespace arma;
using namespace RcppParallel;

// -----------------------------------------------------------------------------
// WORKER 1: Stage 1 & 2 Gradient
// -----------------------------------------------------------------------------

struct Stage12GradientWorker : public Worker {
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
    const vec& eb_theta; // Used if adaptive=1
    const vec& eb_sd;   // Used if adaptive=1

    // Outputs
    vec gradient;

    // Internal dimensions
    int n_fixed, n_random, n_params, nq;

    // Constructor
    Stage12GradientWorker(const vec& params, const mat& X, const mat& U, const mat& Z,
                          const vec& y, const uvec& id, const uvec& unique_ids,
                          const vec& points, const vec& weights, int adaptive,
                          const vec& eb_theta, const vec& eb_sd)
        : params(params), X(X), U(U), Z(Z), y(y), id(id), unique_ids(unique_ids),
          points(points), weights(weights), adaptive(adaptive),
          eb_theta(eb_theta), eb_sd(eb_sd) 
    {
        n_fixed = X.n_cols;
        n_random = U.n_cols;
        n_params = params.n_elem;
        nq = points.n_elem;
        gradient = zeros(n_params);
    }

    // Split constructor
    Stage12GradientWorker(const Stage12GradientWorker& w, Split)
        : params(w.params), X(w.X), U(w.U), Z(w.Z), y(w.y), id(w.id),
          unique_ids(w.unique_ids), points(w.points), weights(w.weights),
          adaptive(w.adaptive), eb_theta(w.eb_theta), eb_sd(w.eb_sd)
    {
        n_fixed = w.n_fixed;
        n_random = w.n_random;
        n_params = w.n_params;
        nq = w.nq;
        gradient = zeros(n_params);
    }

    // The parallel loop
    void operator()(std::size_t begin, std::size_t end) {
        
        // Extract parameter subvectors once per thread
        vec beta = params.subvec(0, n_fixed - 1);
        vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
        vec tau = params.subvec(n_fixed + n_random, n_params - 1);

        for (std::size_t i = begin; i < end; i++) {
            uvec idx = find(id == unique_ids(i));

            mat X_i = X.rows(idx);
            mat U_i = U.rows(idx);
            mat Z_i = Z.rows(idx);
            vec y_i = y.elem(idx);

            // Pre-calculate subject-specific terms
            vec U_alpha_i = U_i * alpha;
            vec Z_tau_i = Z_i * tau;
            vec X_beta_i = X_i * beta;
            vec sigma_v_sq_i = exp(U_alpha_i);
            vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
            vec sigma_epsilon_sq_i = exp(Z_tau_i);
            vec sigma_epsilon_sq_inv_i = 1.0 / sigma_epsilon_sq_i;

            // Adaptive quadrature setup
            vec current_points = points;
            vec current_weights = weights;

            if (adaptive == 1 && i < eb_theta.n_elem && i < eb_sd.n_elem) {
                current_points = eb_theta(i) + eb_sd(i) * points;
                current_weights = eb_sd(i) * exp(0.5 * (points % points - current_points % current_points)) % weights;
            }

            // Accumulators
            double h_i = 0.0;
            vec d_i = zeros(n_params);

            for (int q = 0; q < nq; q++) {
                double theta_1i = current_points(q);
                double w = current_weights(q);

                vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
                vec epsilon = y_i - mu;

                vec epsilon_sigma_ratio = epsilon % sigma_epsilon_sq_inv_i;
                vec epsilon_sq_sigma_ratio = square(epsilon) % sigma_epsilon_sq_inv_i;

                // Gradient components
                vec grad_beta = -2.0 * X_i.t() * epsilon_sigma_ratio;
                vec grad_alpha = -U_i.t() * (epsilon_sigma_ratio * theta_1i % sqrt_sigma_v_sq_i);
                vec grad_tau = Z_i.t() * (1.0 - epsilon_sq_sigma_ratio);

                vec grad_li = join_vert(grad_beta, grad_alpha, grad_tau);
                grad_li *= -0.5;

                const double log_2pi = std::log(2.0 * M_PI);
                vec f_i = log_2pi + log(sigma_epsilon_sq_i) + epsilon_sq_sigma_ratio;
                double l_i = exp(-0.5 * sum(f_i));

                h_i += l_i * w;
                d_i += grad_li * l_i * w;
            }

            if (h_i > 1e-100) {
                gradient += d_i / h_i;
            }
        }
    }

    // Combine results from different threads
    void join(const Stage12GradientWorker& rhs) {
        gradient += rhs.gradient;
    }
};

//' Calculate gradient for Stage 1 & 2 log-likelihood (Fixed Quadrature)
//'
//' Computes the gradient of the log-likelihood for the mixed effects location-scale model
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
//' @return A vector containing the gradient of the log-likelihood.
//' @keywords internal
// [[Rcpp::export]]
arma::vec stage12_gradient_rcpp(const arma::vec& params,
                             const arma::mat& X,
                             const arma::mat& U,
                             const arma::mat& Z,
                             const arma::vec& y,
                             const arma::uvec& id,
                             const arma::vec& points,
                             const arma::vec& weights) {
    uvec unique_ids = unique(id);
    vec dummy_eb; // Placeholder
    
    Stage12GradientWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, 0, dummy_eb, dummy_eb);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.gradient;
}

//' Calculate gradient for Stage 1 & 2 log-likelihood (Adaptive Quadrature)
//'
//' Computes the gradient of the log-likelihood for the mixed effects location-scale model
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
//' @return A vector containing the gradient of the log-likelihood.
//' @keywords internal
// [[Rcpp::export]]
arma::vec stage12_gradient_adaptive(const arma::vec& params,
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
    
    Stage12GradientWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, adaptive, eb_theta, eb_sd);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.gradient;
}


// -----------------------------------------------------------------------------
// WORKER 2: Stage 3 Gradient (Both Fixed and Adaptive)
// -----------------------------------------------------------------------------

struct Stage3GradientWorker : public Worker {
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
    vec gradient;

    // Internal
    int n_fixed, n_random, n_bs, n_params, nq;

    // Constructor
    Stage3GradientWorker(const vec& params, const mat& X, const mat& U, const mat& Z,
                         const vec& y, const uvec& id, const uvec& unique_ids,
                         const vec& points, const vec& weights, const std::string& model_type,
                         int adaptive, const mat& eb_theta2, const mat& eb_thetav)
        : params(params), X(X), U(U), Z(Z), y(y), id(id), unique_ids(unique_ids),
          points(points), weights(weights), model_type(model_type),
          adaptive(adaptive), eb_theta2(eb_theta2), eb_thetav(eb_thetav)
    {
        n_fixed = X.n_cols;
        n_random = U.n_cols;
        n_bs = Z.n_cols;
        n_params = params.n_elem;
        nq = points.n_elem;
        gradient = zeros(n_params);
    }

    // Split constructor
    Stage3GradientWorker(const Stage3GradientWorker& w, Split)
        : params(w.params), X(w.X), U(w.U), Z(w.Z), y(w.y), id(w.id),
          unique_ids(w.unique_ids), points(w.points), weights(w.weights),
          model_type(w.model_type), adaptive(w.adaptive),
          eb_theta2(w.eb_theta2), eb_thetav(w.eb_thetav)
    {
        n_fixed = w.n_fixed;
        n_random = w.n_random;
        n_bs = w.n_bs;
        n_params = w.n_params;
        nq = w.nq;
        gradient = zeros(n_params);
    }

    void operator()(std::size_t begin, std::size_t end) {
        
        const double log_2pi = std::log(2.0 * M_PI);
        vec beta = params.subvec(0, n_fixed - 1);
        vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
        vec tau = params.subvec(n_fixed + n_random, n_fixed + n_random + n_bs - 1);
        vec sigma = params.subvec(n_fixed + n_random + n_bs, n_params - 1);

        // Pre-allocate theta vector
        vec theta;
        if (model_type == "independent") { theta = vec(1); }
        else if (model_type == "linear") { theta = vec(2); }
        else { theta = vec(3); } // interaction or quadratic

        for (std::size_t i = begin; i < end; i++) {
            uvec idx = find(id == unique_ids(i));
            
            mat X_i = X.rows(idx);
            mat U_i = U.rows(idx);
            mat Z_i = Z.rows(idx);
            vec y_i = y.elem(idx);
            
            vec X_beta_i = X_i * beta;
            vec Z_tau_i = Z_i * tau;
            vec U_alpha_i = U_i * alpha;
            vec sigma_v_sq_i = exp(U_alpha_i);
            vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);

            // Adaptive Logic
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
            vec d_i = zeros(n_params);

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
                    else { theta(0) = theta_1i; theta(1) = theta_1i * theta_1i; theta(2) = theta_2i; } // quadratic

                    vec sigma_epsilon_sq = exp(Z_tau_i + dot(sigma, theta));
                    vec sigma_epsilon_sq_inv = 1.0 / sigma_epsilon_sq;

                    vec epsilon_sigma_ratio = epsilon % sigma_epsilon_sq_inv;
                    vec epsilon_sq_sigma_ratio = square(epsilon) % sigma_epsilon_sq_inv;

                    vec grad_beta = -2.0 * X_i.t() * epsilon_sigma_ratio;
                    vec grad_alpha = -U_i.t() * (epsilon_sigma_ratio * theta_1i % sqrt_sigma_v_sq_i);
                    vec grad_tau = Z_i.t() * (1.0 - epsilon_sq_sigma_ratio);

                    vec grad_sigma = zeros(theta.n_elem);
                    double sum_ones_minus_ratio = sum(1.0 - epsilon_sq_sigma_ratio);
                    for(unsigned int k=0; k < theta.n_elem; ++k) {
                        grad_sigma(k) = sum_ones_minus_ratio * theta(k);
                    }

                    vec grad_li = join_vert(grad_beta, grad_alpha, grad_tau, grad_sigma);
                    grad_li *= -0.5;

                    vec f_i = log_2pi + log(sigma_epsilon_sq) + epsilon_sq_sigma_ratio;
                    double l_i = exp(-0.5 * sum(f_i));

                    h_i += l_i * w;
                    d_i += grad_li * l_i * w;
                }
            }

            if (h_i > 1e-100) {
                gradient += d_i / h_i;
            }
        }
    }

    void join(const Stage3GradientWorker& rhs) {
        gradient += rhs.gradient;
    }
};

//' Calculate gradient for Stage 3 (Fixed Quadrature)
//'
//' Computes the gradient of the log-likelihood for the Stage 3 model (bivariate random effects)
//' using standard Gaussian quadrature.
//'
//' @param params A vector of model parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//' @param model_type String specifying the model type ("independent", "linear", "quadratic" and "interaction").
//'
//' @return A vector containing the gradient of the log-likelihood.
//' @keywords internal
// [[Rcpp::export]]
arma::vec stage3_gradient_combined(const arma::vec& params,
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
    
    Stage3GradientWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, model_type, 0, dummy_eb_mat, dummy_eb_mat);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.gradient;
}

//' Calculate gradient for Stage 3 (Adaptive Quadrature)
//'
//' Computes the gradient of the log-likelihood for the Stage 3 model (bivariate random effects)
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
//' @return A vector containing the gradient of the log-likelihood.
//' @keywords internal
// [[Rcpp::export]]
arma::vec stage3_gradient_adaptive_combined(const arma::vec& params,
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
    
    Stage3GradientWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, model_type, adaptive, eb_theta2, eb_thetav);
    parallelReduce(0, unique_ids.n_elem, worker);
    
    return worker.gradient;
}