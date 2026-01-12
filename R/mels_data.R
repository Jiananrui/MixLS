# R/mels_data.R

#' Positive Mood Data
#'
#' A longitudinal dataset containing positive mood measurements and associated variables.
#' This dataset includes 17,514 observations across multiple individuals.
#'
#' @format A data frame with 17,514 rows and 4 variables:
#' \describe{
#'   \item{id}{Integer. Unique identifier for each individual}
#'   \item{posmood}{Numeric. Positive mood score measurement}
#'   \item{alone}{Integer. Binary indicator (0/1) for whether the individual was alone}
#'   \item{genderf}{Integer. Binary indicator (0/1) for gender (likely 1 = female, 0 = male)}
#' }
#' @source research study
#' @examples
#' \dontrun{
#' data(posmood)
#' head(posmood)
#' summary(posmood)
#' }
"posmood"

#' Riesby Depression Data
#'
#' A longitudinal dataset containing Hamilton Depression Rating Scale scores and related variables.
#' This dataset includes 375 observations from a depression study.
#'
#' @format A data frame with 375 rows and 5 variables:
#' \describe{
#'   \item{id}{Integer. Unique identifier for each individual}
#'   \item{hamdep}{Integer. Hamilton Depression Rating Scale score}
#'   \item{week}{Integer. Week of measurement/treatment}
#'   \item{endog}{Integer. Binary indicator (0/1) for endogenous depression}
#'   \item{endweek}{Integer. Interaction term or endpoint week measurement}
#' }
#' @source research study
#' @examples
#' \dontrun{
#' data(riesby)
#' head(riesby)
#' summary(riesby)
#' }
"riesby"
