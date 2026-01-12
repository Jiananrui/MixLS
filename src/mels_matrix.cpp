// src/mels_matrix.cpp

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Solve linear system using appropriate numerical method
//' 
//' Attempts to solve Ax=b using Cholesky decomposition first,
//' falls back to QR decomposition if Cholesky fails
//'
//' @param hess_adj The adjusted Hessian matrix
//' @param grad The gradient vector
//' @return Solution vector, or empty vector if solving fails
// [[Rcpp::export]]
arma::vec solve_matrix_rcpp(const arma::mat& hess_adj, const arma::vec& grad) {
    using namespace arma;
    
    vec delta;
    
    try {
        // Try Cholesky first
        delta = solve(hess_adj, grad, solve_opts::fast + solve_opts::likely_sympd);
    } catch(...) {
        try {
            // If Cholesky fails, try QR decomposition
            delta = solve(hess_adj, grad, solve_opts::equilibrate);
        } catch(...) {
            // If both methods fail, return empty vector to signal error
            return vec();
        }
    }
    
    return delta;
}