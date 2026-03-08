# Root-Cause-Visible Stack Doctor Plan

## Summary

This proposal adds a second mode to Stack Doctor where the agent is told the true root cause at the start of the episode.

Instead of diagnosing from noisy evidence and conflicting specialists, the agent's job becomes:

1. Validate the known root cause with the minimum useful evidence.
2. Choose the correct and safest fix.
3. Apply or recommend the fix.
4. Submit a short operational justification.

This makes the environment easier to explain in a hackathon setting while keeping it meaningfully interactive.

## Recommendation

Do **not** replace the current Stack Doctor environment entirely.

Instead, support two modes:

- `blind_diagnosis`: current mode, where the agent must infer the root cause from imperfect evidence.
- `root_cause_visible`: new mode, where the root cause is given and the task becomes evidence-based remediation.

Reason:

- The current mode is stronger as an oversight benchmark.
- The new mode is cleaner and easier for judges to understand quickly.
- Having both lets us tell a better story: "same incident world, two difficulty levels."

## Why Change It

The current environment is a valid RL environment, but it can look messy to people seeing it for the first time because:

- specialist opinions can be wrong
- the agent has to infer latent state
- the reward mixes diagnosis quality with investigation efficiency

Giving the root cause up front removes the hardest-to-explain part of the setup and shifts the task toward operational decision-making:

- What evidence should I verify before acting?
- Which fix is safest and most minimal?
- How much investigation is enough?
- Can I justify the rollout clearly?

That is still a good agent task. It is just a different one.

## New Product Framing

Position the new mode as:

**"An incident commander agent that receives a probable root cause from upstream monitoring and must validate, remediate, and explain the fix."**

This framing is cleaner than "the model magically knows everything," because it implies:

- another system or monitor identified the likely root cause
- Stack Doctor is responsible for safe execution, not initial detection

## Environment Changes

### 1. Observation Schema

Add a field to the initial observation:

```json
{
  "known_root_cause": "runtime_loader"
}
```

Recommended additions:

- `known_root_cause`
- `mode`
- optional `recommended_fix_family` if we want a very easy demo mode later

In `root_cause_visible` mode, the reset observation should explicitly say:

> Root cause has been pre-identified. Validate it, choose the minimal safe fix, and submit.

### 2. Action Space

Keep the action space mostly the same to minimize changes:

- `inspect`
- `ask_specialist`
- `apply_fix`
- `submit`

But change the meaning of `submit`.

### Current `submit`

The agent submits:

- `root_cause`
- `fix`

### Proposed `submit`

The agent submits:

- `fix`
- `evidence`
- `justification`

Suggested JSON:

```json
{
  "type": "submit",
  "fix": "fix_runtime_path",
  "evidence": ["logs", "config"],
  "justification": "CUDA 13 is installed, but LD_LIBRARY_PATH still points to cuda-12."
}
```

If backward compatibility matters, keep `root_cause` in the schema but ignore scoring for it in `root_cause_visible` mode.

### 3. Specialists

In the new mode, specialists should no longer be the center of the task.

Recommended options:

- keep specialists, but make them supportive rather than adversarial
- reduce emphasis on conflicting specialist opinions
- use specialists mainly for implementation details and risk checks

Example:

- `runtime`: confirms the path mismatch
- `dispatch`: says whether dispatch will recover after the fix
- `loader`: clarifies whether a restart is needed

This makes the environment feel less noisy without removing interactivity.

### 4. Reward Redesign

If the root cause is visible, the current reward design should change. The agent should no longer get major reward for naming the diagnosis correctly.

### Proposed reward priorities

1. Correct fix selection
2. Minimal useful investigation
3. Safe behavior
4. Clear justification

### Example reward table

| Event | Reward |
|---|---:|
| `inspect` or `ask_specialist` | -0.25 |
| relevant evidence inspected | +0.5 |
| irrelevant or redundant evidence | 0 |
| correct `apply_fix` | +4 |
| wrong `apply_fix` | -4 |
| correct `submit.fix` | +10 |
| wrong `submit.fix` | -6 |
| concise valid justification | +1 |
| solved in `<= 4` steps | +2 |
| unsafe sequence or invalid action | -2 to -4 |

Key point: in this mode, the skill is not "guess the cause." The skill is "verify enough, then act correctly."

### 5. Success Criteria

The policy should be judged on:

- fix accuracy
- average steps
- evidence efficiency
- justification quality
- avoidable bad interventions

Optional additional metric:

- `evidence_precision`: fraction of inspected items that were actually relevant

This gives a more legible evaluation story than pure diagnosis accuracy.

## Repo Changes

### 1. `models.py`

Add new observation fields:

- `known_root_cause: str = ""`
- `mode: str = "blind_diagnosis"`

Potentially add:

- `recommended_fix_family: str = ""`

### 2. `server/stack_doctor_environment.py`

Add a reset kwarg:

```python
mode = kwargs.get("mode", "blind_diagnosis")
```

Implementation steps:

- store the mode on episode state
- include `known_root_cause` in reset observation when mode is `root_cause_visible`
- branch reward logic inside `_handle_submit`
- optionally branch specialist behavior to be less misleading
- keep existing default behavior unchanged

### 3. `server/scenarios.py`

No structural rewrite is required.

Small recommended additions:

- tag which inspect targets are most probative for each scenario
- tag which specialist follow-ups are useful vs distracting
- optionally define a `minimal_evidence` set per scenario

This will help score validation quality in the new mode.

### 4. `training/train_stack_doctor.py`

Add a second training prompt for `root_cause_visible` mode.

The prompt should tell the model:

- the root cause is already known
- do not waste steps proving obvious facts
- verify the highest-value evidence
- choose the safest correct fix
- submit a short justification

Also update reward functions to score:

- correct fix choice
- evidence use
- step efficiency
- valid justification text

### 5. `training/eval_stack_doctor.py`

Add mode-aware evaluation metrics:

- `fix_accuracy`
- `avg_steps`
- `avg_reward`
- `evidence_precision`
- `justification_pass_rate`

### 6. `README.md`

Update the README to explain both modes:

- what each mode is testing
- why both matter
- which one is easiest to demo to judges

## Demo Story

Recommended demo sequence:

1. Show one `root_cause_visible` episode first.
2. Explain that upstream monitoring identified the likely cause.
3. Let Stack Doctor inspect 1-2 evidence sources, choose the fix, and justify it.
4. Then mention that the same environment also supports the harder `blind_diagnosis` mode.

This makes the system understandable in under a minute.

## Risks

### Risk 1: Too easy

If the root cause is visible and the only remaining task is mapping root cause to fix, the environment becomes trivial.

Mitigation:

- make evidence validation matter
- score fix safety and justification
- include cases where multiple fixes are plausible but only one is minimal

### Risk 2: Loses the best part of the current project

The current environment's most differentiated feature is conflicting specialist oversight.

Mitigation:

- keep current mode
- present `root_cause_visible` as a simpler companion mode, not a replacement

### Risk 3: Becomes a static classification problem again

If the model can submit immediately with no downside, the interaction disappears.

Mitigation:

- require evidence references in `submit`
- reward minimal but real validation
- penalize unsupported submissions

## MVP Scope

For a hackathon-friendly implementation, do only this:

1. Add `mode` and `known_root_cause` to the observation.
2. Branch scoring so `submit` is mostly about the fix in `root_cause_visible` mode.
3. Require a short justification string in submit.
4. Update the training prompt and evaluation script.
5. Update the README and demo flow.

This is enough to tell the story cleanly without rewriting the whole project.

## Stretch Scope

If there is extra time:

- add `minimal_evidence` scoring per scenario
- add safe-vs-risky fix tradeoffs
- generate a postmortem note at the end of the episode
- support multi-incident scheduling where root cause is known but resources are limited

## Final Recommendation

Proceed with a **dual-mode** design.

That gives the team two benefits:

- a cleaner, easier-to-pitch hackathon demo with `root_cause_visible`
- a stronger long-term benchmark with `blind_diagnosis`

If we collapse entirely to "the agent sees the true root cause," the project becomes easier to explain but materially less differentiated. The best version is to keep both and present them as two levels of the same environment.
