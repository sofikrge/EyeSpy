#!/usr/bin/env Rscript
# Fits one mixed model and writes its tests out as a tidy CSV.
#
# Deliberately dumb: it takes a prepared data file, a formula and an optimizer, fits
# exactly one model, and reports numbers. No filtering, no correction, no decisions --
# those live in CrossPhaseModel.py, the only caller. lme4 + lmerTest are the same engine
# jamovi's GAMLj runs on, which is why the numbers match it.
#
# Usage: Rscript _lmer.R <input.csv> <output.csv> <formula> <optimizer> <factor_coding>
#
# Reads:  a filtered copy of NSS_CrossPhase_LongFormat.csv  (CrossPhaseModel.py)
# Writes: a tidy CSV of omnibus tests, simple effects and fit diagnostics

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

arguments     <- commandArgs(trailingOnly = TRUE)
input_csv     <- arguments[1]
output_csv    <- arguments[2]
model_formula <- as.formula(arguments[3])
optimizer     <- arguments[4]
factor_coding <- arguments[5]

# Set before fitting, because it decides how the factors enter the design matrix.
# "sum" is what jamovi's GAMLj uses; "treatment" is plain R's default.
options(contrasts = c(switch(factor_coding,
                             sum       = "contr.sum",
                             treatment = "contr.treatment",
                             stop("factor_coding must be 'sum' or 'treatment'")),
                      "contr.poly"))

data <- read.csv(input_csv, stringsAsFactors = TRUE)
data$Participant <- factor(data$Participant)   # numeric IDs would become a covariate
data$Image       <- factor(data$Image)

# lmer silently drops incomplete rows. Dropping them here instead keeps model.matrix()
# row-aligned with `data`, which the contrasts below rely on.
data <- data[complete.cases(data[, all.vars(model_formula)]), ]

model <- lmer(model_formula, data = data, REML = TRUE,
              control = lmerControl(optimizer = optimizer))

# One row of the output CSV. Every test fills in only the columns that apply to it.
result_row <- function(table, term, estimate = NA, standard_error = NA,
                       statistic = NA, df = NA, df_residual = NA, p_value = NA) {
  data.frame(table, term, estimate, standard_error, statistic, df, df_residual, p_value)
}

# --- Omnibus F tests, with Satterthwaite denominator df (lmerTest's anova method)
omnibus <- anova(model)
results <- result_row("omnibus", rownames(omnibus),
                      statistic   = omnibus[["F value"]],
                      df          = omnibus[["NumDF"]],
                      df_residual = omnibus[["DenDF"]],
                      p_value     = omnibus[["Pr(>F)"]])

# --- Simple effect of ReferenceMap within each Awareness level (Intact - Scrambled)
# The contrast is read straight off the model matrix: the mean design row of this group's
# Intact cells minus the mean design row of its Scrambled cells. That is what emmeans
# computes, without needing emmeans, and it stays correct if the fixed effects change.
design_matrix <- model.matrix(model)
for (group in levels(data$Awareness)) {
  is_intact    <- data$Awareness == group & data$ReferenceMap == "Intact"
  is_scrambled <- data$Awareness == group & data$ReferenceMap == "Scrambled"
  contrast_weights <- colMeans(design_matrix[is_intact, , drop = FALSE]) -
                      colMeans(design_matrix[is_scrambled, , drop = FALSE])
  contrast <- contest1D(model, contrast_weights)   # 1-df test, Satterthwaite df
  results <- rbind(results, result_row(
    "simple_effect", group,
    estimate       = contrast[["Estimate"]],
    standard_error = contrast[["Std. Error"]],
    statistic      = contrast[["t value"]],
    df_residual    = contrast[["df"]],
    p_value        = contrast[["Pr(>|t|)"]]))
}

# --- Fit diagnostics, reported rather than hidden. A singular fit moves the df around,
# and the log-likelihood is how you spot an optimizer that stopped somewhere worse.
results <- rbind(results, result_row(
  "fit", c("n_observations", "is_singular", "log_likelihood"),
  estimate = c(nobs(model), as.numeric(isSingular(model)), as.numeric(logLik(model)))))

write.csv(results, output_csv, row.names = FALSE)
