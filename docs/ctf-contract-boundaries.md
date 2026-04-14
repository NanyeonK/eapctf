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

### Benchmark anchor
The fixed benchmark anchor is IPCA.
- First objective: reproduce the validated IPCA benchmark path
- Second objective: add uncertainty without disturbing benchmark comparability
- Any uncertainty proposal must be judged against the same fixed IPCA anchor

### Architectural rule
We are not using the old two-stage post-hoc `q_hat` attachment as the active rebuild design.

Required interface for every fitted research model:
- `fit(X, y)`
- `predict_with_uncertainty(X) -> point, uncertainty`

This means point prediction and uncertainty are produced together by one fitted object at prediction time.

### First uncertainty candidates on top of IPCA
- Exposure instability
- Factor reconstruction error
- Rolling residual volatility

### Evaluation rule
For every experiment, persist two outputs separately:
1. Plain benchmark point output
2. Joint point+uncertainty output

Then report:
- benchmark Sharpe
- uncertainty-aware Sharpe
- delta vs benchmark
- exact uncertainty object used

No research claim is valid unless the improvement is stated relative to the fixed benchmark path.

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
- which uncertainty object was used
- whether it was emitted jointly with point prediction
- whether it improved or degraded the fixed IPCA benchmark

## 5. Immediate next runnable path
1. Keep IPCA parity script as the benchmark check
2. Implement the first unified IPCA model object with `predict_with_uncertainty`
3. Start with one conservative uncertainty object
4. Produce benchmark-vs-joint comparison artifacts under the same metric path
