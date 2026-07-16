# GPT-5.6-Sol-Pro evaluation prompt: general hallucination mitigation for SNVLA

## Prompt

You are reviewing a proposed hallucination-mitigation design for a
vision-language-action robot policy. Treat this prompt as the complete source of context. You do
not have access to a prior conversation, hidden design document, private training log, or unstated
requirement. You may inspect the GitHub repository and immutable revision explicitly linked below;
do not assume access to any other source. If information needed for a firm conclusion is missing,
identify it explicitly and state how it should be measured.

Your task is to critically evaluate the proposal, find weaknesses that its authors may have missed,
compare it with credible alternatives, and recommend a concrete experimental and implementation
plan. Do not merely agree with the proposal. Prefer mechanisms that generalize across many robot
tasks rather than mechanisms specialized for one block-picking environment.

### Repository access

The implementation is available in the following GitHub repository:

- Repository: <https://github.com/0xNOY/lerobot-policy-snvla>
- Review branch: <https://github.com/0xNOY/lerobot-policy-snvla/tree/feat/p5-e2-sim-eval>
- Immutable review revision:
  <https://github.com/0xNOY/lerobot-policy-snvla/tree/1d987ca46b1784caf8023d8afdb974656efa9b8a>
- Commit SHA: `1d987ca46b1784caf8023d8afdb974656efa9b8a`

Inspect the code at the immutable revision before finalizing the review. In particular, examine:

- `src/lerobot_policy_snvla/modeling_snvla.py` for the joint action/narration model, losses,
  narration generation, and history handling;
- `src/lerobot_policy_snvla/processor_snvla.py` for prompt construction, state dropout, observation
  noise, and narration targets;
- `src/lerobot_policy_snvla/configuration_snvla.py` for available configuration and inference
  behavior;
- `src/lerobot_policy_snvla/training_schedule.py` for augmentation schedules;
- `src/lerobot_policy_snvla/sim/events.py` and `src/lerobot_policy_snvla/sim/evaluate.py` for
  physical-event definitions and false-narration evaluation;
- `src/lerobot_policy_snvla/scripts/prepare_success_dataset.py` and
  `src/lerobot_policy_snvla/scripts/augment_narrations.py` for dataset construction; and
- `docs/superpowers/plans/2026-07-14-p5-e2-success-only-state-dropout.md` plus
  `docs/superpowers/reports/2026-07-14-p5-e2-success-only-report.md` for the implemented experiment
  specification and evidence.

Use code inspection to correct any inaccurate architectural assumption in this prompt. Explicitly
identify each correction and cite the relevant repository path and symbol or line. Distinguish what
is verified from code from what is inferred. If the repository is private or otherwise inaccessible
in your ChatGPT session, state that limitation prominently and continue using only the complete
context provided below; do not pretend that code inspection occurred.

### 1. System being reviewed

The system is called SNVLA. It is a vision-language-action policy initialized from
`lerobot/pi05_base`. It jointly performs two functions:

1. It predicts robot action chunks through an Action Expert.
2. It optionally generates short natural-language narrations describing robot progress and
   physical outcomes.

At inference time, the policy receives:

- one or more current RGB camera observations;
- the task instruction;
- the current robot proprioceptive state;
- previously accepted narrations; and
- internal language-model context.

The Action Expert is conditioned on the robot state and visual representation. The narration path
can choose between a beginning-of-narration mode and a beginning-of-action/no-narration mode, then
generate language tokens. A generated narration is currently appended to `previous_narrations` and
therefore affects later inference. Action and narration are trained jointly with an action diffusion
loss, narration cross-entropy, and mode-selection losses.

The current production configuration uses:

- `max_state_dim=32`;
- `max_action_dim=32`;
- `n_action_steps=40`;
- 500 successful simulator demonstrations, split into 450 training and 50 validation episodes;
- narration augmentation with a forward-only window of 10 frames; and
- initialization from `lerobot/pi05_base`.

### 2. Observed failures

The motivating failures are:

1. **False physical-outcome narration:** the model says that an object was picked up even though
   the gripper failed to grasp it.
2. **False placement narration:** the model says that an object was placed even though it did not
   reach or remain in the target region.
3. **Premature task completion:** the model emits `Task completed.` before the required physical
   goal is satisfied.
4. **Poor manipulation:** the end effector sometimes fails to approach the object. This is an
   action-quality failure and must be distinguished from narration hallucination, although the two
   may interact.
5. **Possible proprioceptive shortcut:** a working hypothesis is that narration generation may use
   robot state or learned temporal priors as a trigger instead of checking the images for physical
   evidence.
6. **History contamination:** once a false narration is appended to `previous_narrations`, later
   narration and action predictions may be conditioned on a false account of the episode.

The initial task was a three-object pick-and-place task, but the desired mitigation must also be
reusable for tasks such as drawer manipulation, button pressing, stacking, pouring, folding,
navigation, tool use, and future tasks with different event vocabularies.

### 3. Mitigations already implemented

Do not present the following as new proposals; they already exist:

1. The production dataset was increased to 500 successful demonstrations.
2. Narration timing was corrected so that final placement occurs before returning to a fixed home
   pose, and `Task completed.` is emitted only after arriving home.
3. Frames more than 10 frames after `Task completed.` are excluded.
4. Forward-only narration augmentation uses a window size of 10.
5. Language-side state dropout is applied to 25% of eligible training frames. On those frames the
   state line is removed from the language prompt, while the Action Expert still receives robot
   state and action loss remains active.
6. A deterministic schedule prevents a frame that had state hidden in one epoch from being hidden
   again in the immediately following epoch.
7. Observation noise is applied to 25% of eligible data. A per-sample noise scale is drawn from
   `0.0` to `0.5`; noise is applied to normalized robot state and all camera images. Action and
   narration losses remain active.
8. Evaluation records false pick, false place, and false task-completion counters, physical success,
   pick/place counts, and minimum end-effector-to-object distance.
9. The simulator has privileged physical event detectors for evaluation and dataset validation,
   but privileged simulator state must not be required by the deployed runtime policy.

These measures may reduce shortcuts or measure hallucination, but they do not directly prove that a
generated narration is causally grounded in current visual evidence.

### 4. Previously rejected or constrained directions

- A corrective-pilot data-collection plan was discontinued because the generated corrective
  behavior had serious quality problems. Do not assume that corrective actions can safely be used
  as action teachers.
- A task-specific runtime gate based directly on simulator object coordinates would not transfer to
  a real robot and is therefore unsuitable as the primary general solution.
- The Action Expert should retain access to robot state. Hallucination mitigation must not remove
  proprioception from action prediction.
- A solution that suppresses all narration is not acceptable. Correct narration recall and useful
  coverage must be preserved.
- Runtime verification should avoid materially slowing the action-control loop. Candidate-only or
  asynchronous verification is allowed if its synchronization semantics are well defined.

Offline simulator state may be used to create labels or evaluate a checkpoint, provided the trained
runtime mechanism consumes only deployable observations.

### 5. Proposed general solution

The proposed design deliberately avoids a fixed `picked / placed / completed` classifier and avoids
a hand-written finite-state machine tied to one task. It has four components.

#### Proposal A: candidate-conditioned temporal visual verifier

Add a verifier that evaluates whether a candidate narration is supported by recent visual evidence:

```text
V(images[t-k:t], candidate_narration) -> supported / uncertain / unsupported
```

Candidate inputs include arbitrary statements such as:

- `Picked up the red block.`
- `The drawer is open.`
- `The cup was placed on the tray.`
- `The button has been pressed.`
- `The robot reached the requested room.`
- `Task completed.`

The verifier should use multiple current/recent camera frames and the candidate text. The initial
proposal excludes robot state, action, and `previous_narrations` so that these cannot become shortcut
triggers. Evaluate whether the verifier should additionally receive the original task instruction,
especially for generic statements such as `Task completed.`, and whether task input introduces a
new language-prior shortcut.

The verifier may be a frozen-backbone lightweight head, a separately trained video-language model,
or a jointly trained module. The choice has not been made.

#### Proposal B: counterfactual multimodal grounding

Train the narration path and/or verifier using positive image-text pairs and automatically generated
hard negatives. Candidate negative construction includes:

- replacing the current frames with earlier frames from the same episode;
- swapping frames across an event boundary;
- pairing the narration with another episode's images;
- masking a task-relevant image region;
- making one camera stale while keeping another current;
- changing the narrated object, state, spatial relation, quantity, or completion claim; and
- moving a completion statement to an incomplete observation.

A possible ranking objective is:

```text
L_ground = max(
    0,
    margin - score(real_images, narration)
           + score(counterfactual_images, narration)
)
```

The intended effect is that a narration must score better with its true visual evidence than with a
plausible but incorrect visual context. The design must address false negatives, artificial cues in
generated negatives, temporal alignment, and cases in which the narration is true across a long
time interval rather than at one frame.

#### Proposal C: calibrated abstention

Use verifier confidence to decide whether to publish a narration:

```text
high support confidence -> publish
low support confidence  -> reject
intermediate confidence -> abstain or defer
```

Calibration should be evaluated using Brier score, expected calibration error, risk-coverage
curves, and event-level precision/recall. A false-positive-sensitive threshold is desired, but a
minimum narration recall or coverage requirement must prevent the trivial always-abstain solution.

Evaluate binary versus three-way verifier outputs, temperature scaling, conformal prediction, and
whether thresholds can generalize across tasks and domains.

#### Proposal D: provisional narration history

Do not immediately append a candidate narration to `previous_narrations`. Keep it provisional until
the verifier accepts it:

```text
candidate generated
    -> verifier accepted    -> publish and commit to history
    -> verifier uncertain   -> defer or discard
    -> verifier rejected    -> discard
```

Raw candidates, verifier scores, decisions, and timestamps should still be recorded for analysis.
The design must specify behavior when verification finishes after the action loop has already
advanced, how to prevent repeated rejected candidates, whether later evidence may validate a
deferred statement, and whether action prediction may consume provisional or only committed
history.

### 6. Candidate system architecture

The current candidate architecture is:

```text
                         +------------------------------+
RGB images + robot state | Action Expert                | -> action chunk
------------------------>| state remains available      |
                         +------------------------------+

RGB images + task text   +------------------------------+
+ committed narration -->| Narration Generator          | -> candidate text
history                  +------------------------------+
                                          |
                                          v
recent multi-camera      +------------------------------+
frames + candidate text ->| Temporal Visual Verifier     |
(possibly task text)     | no robot state/action input  |
                         +------------------------------+
                                | accepted / uncertain / rejected
                                v
                         publish and commit / abstain
```

The action loop should not wait for expensive verification unless a justified synchronization rule
requires it. One possible implementation is to run the verifier only after a narration candidate is
generated and to execute it asynchronously alongside subsequent action inference.

### 7. Possible training and diagnostic measurements

Before changing the production policy, the following causal diagnostic has been proposed:

1. Save narration-event token probabilities or candidate scores for the real observation.
2. Re-run the same candidate with robot state, task, and history held fixed while changing only the
   images to stale, shuffled, cross-episode, or masked versions.
3. Measure:

```text
visual_evidence_gap =
    log p(candidate | real images)
  - log p(candidate | counterfactual images)
```

A small gap for a physical-outcome statement would support the hypothesis that the model does not
depend sufficiently on visual evidence.

Potential evaluation dimensions include:

- false narration rate for physical outcomes;
- correct narration recall and coverage;
- physical task success and action quality;
- event detection latency;
- verifier calibration and selective risk;
- real-versus-counterfactual evidence gap;
- robustness to camera occlusion, stale frames, viewpoint changes, distractors, lighting changes,
  and unseen objects;
- cross-task and cross-scene transfer;
- added inference latency, GPU memory, and throughput; and
- frequency and downstream impact of rejected or deferred narrations.

### 8. Relevant research directions

Use these only as starting points; independently assess whether their assumptions transfer to this
robot setting:

- Hallucination-Augmented Contrastive Learning uses hallucinated text as hard negatives for
  cross-modal alignment: <https://arxiv.org/abs/2312.06968>
- MDPO adds image-conditioned preference optimization because ordinary multimodal preference
  training may ignore the image condition: <https://arxiv.org/abs/2406.11839>
- Robust visual instruction tuning reports benefits from balanced positive and negative visual
  instructions: <https://arxiv.org/abs/2306.14565>
- Visual Contrastive Decoding contrasts predictions under original and distorted images at
  inference time: <https://arxiv.org/abs/2311.16922>
- AHA trains a vision-language model to detect and explain robotic manipulation failures:
  <https://arxiv.org/abs/2410.00371>
- SayCan grounds language decisions with learned physical affordances:
  <https://arxiv.org/abs/2204.01691>
- Inner Monologue studies closed-loop feedback including success detection:
  <https://arxiv.org/abs/2207.05608>
- Act, Think or Abstain studies uncertainty-aware routing for VLA models:
  <https://arxiv.org/abs/2603.05147>

Distinguish findings directly supported by cited work from your own inference. If you have browsing
or literature-search capability, prefer primary papers and include direct links for any additional
claims.

### 9. Required review questions

Answer all of the following:

1. Does the proposed verifier-plus-counterfactual-training design directly address the suspected
   image-ignoring shortcut? Under what failure modes would it not?
2. Is a separate verifier meaningfully safer than the generator, or can both models learn the same
   shortcut? What independence, data, or architectural constraints are needed?
3. Should the verifier receive the task instruction? Should it receive committed narration history?
   Explain the grounding benefit and shortcut risk of each input.
4. Should the verifier consume a single frame, a fixed temporal window, learned memory, frame
   differences, or object-centric tracks? Recommend a concrete default.
5. How should multi-camera disagreement, occlusion, stale frames, and asynchronous timestamps be
   handled?
6. Which negative-pair generation methods are likely to create reliable hard negatives, and which
   are likely to introduce false negatives or synthetic artifacts?
7. Should counterfactual grounding train the main narration generator, only the verifier, or both?
   Could joint training degrade action performance through the shared visual backbone?
8. Which loss formulation is most appropriate: binary cross-entropy, focal loss, Brier loss,
   InfoNCE, margin ranking, unlikelihood loss, DPO-style preference loss, or a combination?
9. How should uncertainty be calibrated under task and domain shift? Are fixed thresholds adequate?
10. How should deferred and rejected narrations interact with action generation and history without
    creating race conditions or feedback loops?
11. Can this design transfer to real robots when training labels were partly generated from
    privileged simulator state? Specify the necessary sim-to-real controls.
12. Compare the proposal with at least three alternatives, including a strong alternative you would
    prefer if you reject the proposal.
13. Identify the smallest experiment that could falsify the central hypothesis or show that the
    verifier adds no value.
14. Define quantitative go/no-go gates that jointly prevent false narration and an always-abstain
    collapse.
15. State whether the four proposals should be accepted, revised, reordered, or rejected.

### 10. Required response format

Produce the review in the following structure:

1. **Executive verdict** — no more than 250 words.
2. **Assumptions and missing evidence** — facts that must be measured rather than assumed.
3. **Proposal-by-proposal assessment** — a table covering A through D, with 1–5 scores for:
   expected hallucination reduction, cross-task generality, data efficiency, runtime cost,
   implementation risk, sim-to-real robustness, and falsifiability. Explain every score briefly.
4. **Critical failure analysis** — shortcut leakage, correlated verifier/generator errors,
   calibration failure, temporal ambiguity, false negatives, action degradation, and history races.
5. **Recommended architecture** — specify verifier inputs, excluded inputs, temporal context,
   backbone-sharing choice, loss functions, calibration, and runtime synchronization.
6. **Minimal decisive experiment** — datasets, splits, baselines, controls, metrics, approximate
   sample requirements, and the result that would falsify the proposal.
7. **Ablation matrix** — at minimum: current baseline; diagnostic only; verifier only;
   counterfactual training only; verifier plus counterfactual training; abstention; provisional
   history; and the complete system.
8. **Go/no-go criteria** — explicit numerical or statistically defined gates. If exact values cannot
   be justified in advance, give a principled calibration procedure rather than inventing numbers.
9. **Implementation sequence** — ordered stages with rollback points and expected evidence from each
   stage.
10. **Final decision** — choose one: accept, accept with revisions, reject, or replace. List the
    minimum required revisions.

Be concrete, skeptical, and technically detailed. Optimize for a solution that remains useful when
the robot task, objects, environment, camera layout, embodiment, and narration vocabulary change.
