# ==============================================================================
# Course    : Econometrics
# Topic     : Binary choice models with gradient descent algorithm
# Instructor: Abdulbaki Bilgic
# ==============================================================================
# Clean Environment:
# ==============================================================================
rm(list = ls(all.names = TRUE))
cat("\014")
graphics.off()
closeAllConnections()
options(scipen = 999, digits = 4)
# ==============================================================================
# Data Generation Process:
# ==============================================================================
set.seed(42)

n <- 500
k <- 3

X           <- matrix(rnorm(n * k), nrow = n, ncol = k)
colnames(X) <- c("X1", "X2", "X3")

true_b <- -0.5
true_w <- c(1.5, -2.0, 0.8)

z <- X %*% true_w + true_b
p <- 1 / (1 + exp(-z))
y <- rbinom(n, size = 1, prob = p)
# ==============================================================================
# Link Functions:
# ==============================================================================
link_function <- function(z, method) {
  if (method == "logit")  return(plogis(z))
  if (method == "probit") return(pnorm(z))
  stop("method must be 'logit' or 'probit'")
}
# ==============================================================================
# Binary Choice Models via Correct Gradient Descent (MLE):
# ==============================================================================
binary_fit <- function(X,
                       y,
                       method   = c("logit", "probit"),
                       lr       = 0.10,
                       max_iter = 10000,
                       tol      = 1e-12) {
  
  method <- match.arg(method)
  
  n   <- nrow(X)
  k   <- ncol(X)
  eps <- 1e-9
  
  b   <- 0
  w   <- rep(0, k)
  
  costs <- numeric()
  
  for (i in seq_len(max_iter)) {
    
    b_old <- b
    w_old <- w
    
    z     <- as.vector(b + X %*% w)
    p_hat <- link_function(z, method)
    
    # Log-likelihood
    ll <- sum(
      y * log(p_hat + eps) +
        (1 - y) * log(1 - p_hat + eps)
    )
    costs <- c(costs, -ll / n)
    
    # Score functions: 
    if (method == "logit") {
      
      db <- mean(p_hat - y)
      dw <- t(X) %*% (p_hat - y) / n
      
    } else if (method == "probit") {
      
      phi <- dnorm(z)
      wgt <- phi / (p_hat * (1 - p_hat) + eps)
      
      db <- -mean(wgt * (y - p_hat))
      dw <- -t(X) %*% (wgt * (y - p_hat)) / n
      
    }
    # gradient descent algorithm:
    b <- b - lr * db
    w <- w - lr * as.vector(dw)
    
    # set the tolarence rule:
    delta <- max(abs(c(b - b_old, w - w_old)))
    if (delta < tol) break
  }
  # ============================================================================
  # Fisher Information Matrix:
  # ============================================================================
  z     <- as.vector(b + X %*% w)
  p_hat <- link_function(z, method)
  
  if (method == "logit") {
    W <- diag(as.vector(p_hat * (1 - p_hat)))
  } else if (method == "probit") {
    phi <- dnorm(z)
    W   <- diag(as.vector(phi^2 / (p_hat * (1 - p_hat) + eps)))
  }
  
  X_ext <- cbind(1, X)
  vcov  <- solve(t(X_ext) %*% W %*% X_ext)
  
  list(
    method    = method,
    b         = b,
    w         = w,
    vcov      = vcov,
    costs     = costs,
    iter      = i,
    converged = (delta < tol)
  )
}
# ==============================================================================
# Summary Function:
# ==============================================================================
binary_summary <- function(model, X) {
  
  beta <- c(model$b, model$w)
  se   <- sqrt(diag(model$vcov))
  zval <- beta / se
  pval <- 2 * (1 - pnorm(abs(zval)))
  
  stars <- cut(
    pval,
    breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
    labels = c("***", "**", "*", ""),
    right  = FALSE
  )
  
  out <- data.frame(
    Estimate  = sprintf("%.4f", beta),
    Std.Error = sprintf("%.4f", se),
    z.value   = sprintf("%.4f", zval),
    p.value   = sprintf("%.4f", pval),
    Signif    = stars,
    stringsAsFactors = FALSE
  )
  
  rownames(out) <- c("(Intercept)", colnames(X))
  return(out)
}
# ==============================================================================
# Model Estimation (Set Your Choice):
# ==============================================================================
method_choice <- "probit"   # set your choice either "logit" or "probit"

model <- binary_fit(
  X      = X,
  y      = y,
  method = method_choice
)

cat("\n================ GD MODEL OUTPUT ================\n")
cat("Method     :", toupper(method_choice), "\n")
cat("Iterations :", model$iter, "\n")
cat("Converged  :", model$converged, "\n")
cat("Final Cost :", sprintf("%.6f", tail(model$costs, 1)), "\n\n")

print(binary_summary(model, X))
# ==============================================================================
# GLM Comparison:
# ==============================================================================
df <- data.frame(y = y, X)

glm_model <- glm(
  y ~ X1 + X2 + X3,
  data   = df,
  family = binomial(link = method_choice)
)

coef_mat <- summary(glm_model)$coefficients

glm_out <- data.frame(
  Variable  = rownames(coef_mat),
  Estimate  = sprintf("%.4f", coef_mat[, 1]),
  Std.Error = sprintf("%.4f", coef_mat[, 2]),
  z.value   = sprintf("%.4f", coef_mat[, 3]),
  p.value   = sprintf("%.4f", coef_mat[, 4]),
  row.names = NULL,
  stringsAsFactors = FALSE
)

cat("\n================ GLM OUTPUT ==================\n")
print(glm_out)
# ==============================================================================