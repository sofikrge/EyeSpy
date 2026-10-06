# PowerAnalysisKumle.R
# Power analysis according to Kumle et al., (2021)
# Following this tutorial: https://lkumle.github.io/power_notebooks/Scenario1_notebook.html

# load libraries
library (lme4)
library (mixedpower)
library(writexl)
library(ggplot2)
library(tidyr)
library(openxlsx)


# load the data
data <- read.csv('/Users/sofiakarageorgiou/Desktop/Replication/Power Analysis/NSS_CrossPhase_LongFormat 2.csv')
data <- data[data$Awareness != "conscious_aware", ] # filter out conscious session, comment out if you want full model with all 3 awareness levels


# Match jamovi's factor coding: "Simple" contrasts
data$Awareness   <- factor(data$Awareness)
data$ReferenceMap <- factor(data$ReferenceMap)
simple_contrast <- function(f) contr.treatment(nlevels(f)) - 1 / nlevels(f)
contrasts(data$Awareness)    <- simple_contrast(data$Awareness)
contrasts(data$ReferenceMap) <- simple_contrast(data$ReferenceMap)

# set parameters 
fixed_effects <- c("ReferenceMap", "Awareness")
simvar <- "Participant" #which random effect do we want to vary in the simulation?
# Simulation parameters
steps <- c(20, 25, 30, 35, 40, 45, 50)

# critical_value <- 1.96 # when only keeping interaction in the treeBH
critical_value <- 2.24 # when adding only main effect of disambiguator
# critical_value <- 2.394 # when adding both main effects (disambiguator and awareness), bc 0.05/3

n_sim <- 1000 # how many simulations 
reduction_factor <- 0.8

# fit model on unconscious dataset
NSSmodel <- lmer(NSS ~ 1 + ReferenceMap + Awareness +
                   ReferenceMap:Awareness +
                   (1 + ReferenceMap | Participant) + (1 + ReferenceMap | Image),
                 data = data, control = lmerControl(optimizer = "bobyqa"))

# SESOI for the full model: shrink every fixed effect (except the intercept) to
# `reduction_factor` (0.70) of its observed value.
SESOI_auto <- fixef(NSSmodel)
SESOI_auto[2:length(SESOI_auto)] <- SESOI_auto[2:length(SESOI_auto)] * reduction_factor

# run power analysis
power_full <- mixedpower(model = NSSmodel, data = data, 
                         fixed_effects = fixed_effects,
                         simvar = simvar, steps = steps,
                         critical_value = critical_value, n_sim = n_sim,
                         SESOI = SESOI_auto, databased = T)


# only for simple effect within unconscious aware
# keep only unconscious aware
data_aware <- data[data$Awareness == "unconscious_aware", ]

# fit simple model so disambiguator type is only fixed effect
NSSmodel_aware <- lmer(NSS ~ 1 + ReferenceMap +
                         (1 | Participant) + (1 | Image),
                       data = data_aware, control = lmerControl(optimizer = "bobyqa"))

# SESOI for the simple-effect model: shrink the fixed effect (except the intercept)
# to 0.70 of its observed value
SESOI_aware <- fixef(NSSmodel_aware)
SESOI_aware[2:length(SESOI_aware)] <- SESOI_aware[2:length(SESOI_aware)] * 0.70


# critical_value_simple <- 2.24  # vs 1.96 for the interaction; doubled-p correction, only if interaction is in TreeBH
critical_value_simple <- 2.5 # when adding disamb main effect to tree bh
# idea is that we have a double correction so 0.05/2/2 = 0.0125 -> critical value of that is 2.5
# critical_value_simple <- 2.638 # when adding both main effects (bc. 0.05/2/3 alpha level)

# ============================================================
power_simple <- mixedpower(model = NSSmodel_aware, data = data_aware,
                           fixed_effects = c("ReferenceMap"),
                           simvar = simvar, steps = steps,
                           critical_value = critical_value_simple, n_sim = n_sim,
                           SESOI = SESOI_aware, databased = T)

# save data as excel file
write.xlsx(as.data.frame(power_full),
           "/Users/sofiakarageorgiou/Desktop/Replication/2406Analysis/Power Analysis/INTERPOLpower.xlsx",
           rowNames = TRUE)
write.xlsx(as.data.frame(power_simple),
           "/Users/sofiakarageorgiou/Desktop/Replication/2406Analysis/Power Analysis/INTERPOLpower_simple.xlsx",
           rowNames = TRUE)


# plot

# Build the plot from BOTH power objects so each line comes from the right model.

# Interaction: the ReferenceMap2:Awareness2 row from the full model.
df_int <- as.data.frame(power_full)
df_int$rowname <- rownames(df_int)
df_int <- df_int[grepl(":", df_int$rowname), ]
df_int$effect_label <- "Interaction"

# Simple effect of Intact vs Scrambled WITHIN unconscious aware: the ReferenceMap
# effect from the model refit on the unconscious_aware subset (power_simple).
# (NOT the ReferenceMap main effect in power_full, which is averaged over awareness.)
df_smp <- as.data.frame(power_simple)
df_smp$rowname <- rownames(df_smp)
df_smp <- df_smp[df_smp$effect == "ReferenceMap2", ]
df_smp$effect_label <- "Simple effect within unconscious aware"

df_plot <- rbind(df_int, df_smp)

# label the mode cleanly
df_plot$mode <- ifelse(df_plot$mode == "databased", "Observed effect", "SESOI (80% of observed)")

# reshape to long format
# Grab whatever participant-count columns actually exist (everything that isn't a
# metadata column) so this stays in sync with `steps` instead of being hardcoded.
step_cols <- setdiff(names(df_plot), c("mode", "effect", "rowname", "effect_label"))
df_long <- pivot_longer(df_plot,
                        cols = all_of(step_cols),
                        names_to = "Participants",
                        values_to = "Power")
df_long$Participants <- as.numeric(df_long$Participants)

# plot
ggplot(df_long, aes(x = Participants, y = Power,
                    color = effect_label, linetype = mode)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0.90, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  annotate("text", x = min(steps) + 1, y = 0.91, label = "", size = 3, color = "gray50") +
  scale_y_continuous(limits = c(0.6, 1.05), breaks = seq(0.4, 1.0, 0.1)) +
  scale_x_continuous(breaks = steps) +
  labs(title = "Power analysis",
       x = "Number of Participants",
       y = "Statistical Power",
       color = "Effect",
       linetype = "Effect Size") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid = element_blank(),
        axis.line = element_line(color = "gray80"))

# save
ggsave("/Users/sofiakarageorgiou/Desktop/Replication/2406Analysis/Power Analysis/INTERPOLpowerplot.png",
       dpi = 300, width = 12, height = 8)


# ---- singular-fit diagnostics ----------------------------------------------
# A singular fit means the data can't support all the random-effects parameters
# (a variance pinned at 0, or an intercept-slope correlation pinned at +/-1).
# If a BASE model is singular, the power estimates built from it are unreliable.
# Want FALSE for both. The simple-effect model (smaller subset) is the likely risk.

cat("\n==== Singular-fit checks ====\n")
cat("Full model    isSingular:", isSingular(NSSmodel), "\n")
cat("Simple model  isSingular:", isSingular(NSSmodel_aware), "\n")

# Inspect the random-effects estimates: look for variances ~0 (Std.Dev near 0)
# or correlations of +/-1.00, which is what drives a singular fit.
cat("\n---- Full model: random-effects (VarCorr) ----\n")
print(VarCorr(NSSmodel), comp = c("Variance", "Std.Dev."))
cat("\n---- Simple model: random-effects (VarCorr) ----\n")
print(VarCorr(NSSmodel_aware), comp = c("Variance", "Std.Dev."))