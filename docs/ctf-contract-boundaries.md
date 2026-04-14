# CTF contract boundaries

This document separates three layers that must not be mixed.

## 1. Fixed CTF contract
These are frozen by the competition and are not research choices.

### Inputs
The server provides the same input contract to every participant.
- `chars`
- `features`
- `daily_ret`

Interpretation:
- `chars`: stock-by-month-end characteristic panel
- `features`: allowed feature list
- `daily_ret`: daily return panel used for next-month portfolio evaluation

### Submission interface
Participants must submit:
- `main(chars, features, daily_ret) -> DataFrame`

Returned columns must be exactly:
- `id`
- `eom`
- `w`

The server does not evaluate raw predictions or uncertainty objects directly. It evaluates only the final portfolio weights.

### Test scope
- Test months are fixed by `ctff_test`
- Every flagged test month must be covered

### Server metric
The server fixes the final leaderboard metric.
- Portfolio volatility is scaled to 10% annualized target
- Leaderboard ranking uses Sharpe ratio after that scaling
- Return aggregation is based on the next month from `daily_ret`

### Compliance rules
These are part of the fixed contract.
- No look-ahead bias
- No missing weights on required test rows
- No duplicate `(id, eom)` rows
- No NaN weights
- No external network dependency
- Deterministic output for the same inputs

## 2. Participant choice contract
These are the legitimate strategy degrees of freedom inside the fixed CTF shell.

### Allowed choices
- Feature subset choice
- Feature preprocessing and normalization
- Model class
- Hyperparameter tuning
- Training window design
- Refit schedule
- Signal-to-weight mapping
- Whether and how uncertainty enters weight construction

### Not allowed
- Changing the server metric
- Changing the test period definition
- Returning a different submission schema
- Using future information

## 3. Our research contract
This project adds one more layer on top of the participant choice set.

### Benchmark anchor vs architecture target
IPCA should be treated as the first audited rebuild anchor, not as the permanent sole baseline for the package.

Meaning:
- First objective: reproduce the validated IPCA benchmark path
- Second objective: use that audited path to lock metric parity and file/contract conventions
- Third objective: extend the same joint interface to many model families

So there are two different ideas:
1. `IPCA` as the first anchor for rebuild parity
2. `model-agnostic joint interface` as the permanent package architecture

### Architectural rule
We are not using the old two-stage post-hoc `q_hat` attachment as the active rebuild design.

Required interface for every fitted research model:
- `fit(X, y)`
- `predict_with_uncertainty(X) -> point, uncertainty`

This means point prediction and uncertainty are produced together by one fitted object at prediction time.

### Model-family rule
`eapctf` should allow multiple model families through the same contract.
Initial families should include:
- IPCA
- OLS / linear models
- tree-based ML
- neural-network models

The package should not hard-code a permanent assumption that only IPCA is valid.

### Evaluation rule
For every experiment, persist two outputs separately:
1. Plain point baseline for that same model family
2. Joint point+uncertainty output for that same model family

Then report at least two deltas:
- delta versus the same model's point-only baseline
- delta versus the fixed IPCA anchor

This keeps both comparisons visible:
- "Does uncertainty help this model?"
- "Does this model family beat the audited anchor?"

### First uncertainty candidates
- Exposure instability
- Factor reconstruction error
- Rolling residual volatility

## 4. Operational implication for `eapctf`
`eapctf` should encode these boundaries explicitly.

### Frozen layer
Keep fixed and auditable:
- input names
- submission columns
- test flag source
- CTF evaluation logic
- compliance checks

### Variable layer
Allow swapping:
- model implementation
- feature handling
- uncertainty object
- portfolio weighting rule inside the submission function

### Research layer
Require explicit reporting:
- which model family was used
- which uncertainty object was used
- whether uncertainty was emitted jointly with point prediction
- whether it improved the model-specific point baseline
- whether it improved or degraded the IPCA anchor

## 5. Immediate next runnable path
1. Keep IPCA parity script as the anchor check
2. Generalize the joint interface so non-IPCA families can plug in cleanly
3. Benchmark each model family first in point-only form
4. Add joint uncertainty variants and compare against both the model-specific baseline and the IPCA anchor
