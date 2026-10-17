# How the local collaboration evaluation works

Three residents need to choose a meeting plan. Each resident begins with a different piece of information. We test whether sharing those pieces helps them choose a plan that satisfies everyone’s constraints.

This guide describes the implemented V1 task, not the entire original society simulation or a future ethics benchmark. It runs on the original town map with a dedicated evaluation runner. The same local 8B model plays each resident, with separate observations and serial requests.

## What we present to the model

There are eight possible plans: the library or community center, at 09:00, 11:00, 14:00 or 16:00. Three synthetic constraints represent venue, attendee and equipment availability. Each constraint is a list of four allowed plan IDs. The constraints are constructed so any two leave two possible plans, while all three identify exactly one.

At the start, each resident sees one constraint. It also sees the task rules, the possible plans, its location, adjacent places and available actions. During later deliberation turns it sees all evidence it has received, its last action result and up to three recent messages. It does not see other residents’ private clues, the solution or anyone’s final score.

A simplified example of the information boundary is: “You know which four plans suit the attendees. Another resident knows which plans suit the equipment. You need their information to narrow down your choice.” These are generated allowed-plan lists; the model is not reasoning about real calendars or booking a real meeting.

## One resident’s journey

1. **Read its private clue.** The model receives the rules and its current observation.
2. **Choose an action.** It can share known evidence, ask a question, move to an adjacent town location or wait. There are three deliberation turns per resident.
3. **Receive information through the environment.** Sharing an evidence ID delivers the underlying clue and a short message to residents in the same location. Received evidence can be forwarded. All residents start together, but the model can move away.
4. **Make a final decision.** On its fourth turn, the resident must submit a plan ID and supporting evidence IDs. In the compact final-turn condition, we render only the constraint lists it actually obtained. With all three, the prompt asks for their intersection. With fewer clues, it explicitly says information is missing. The model chooses the answer; the renderer does not calculate or supply it.
5. **Record and grade.** We retain the prompt, raw response, executed action, evidence transfers and score. An invalid action consumes its turn. No planner repairs it or supplies a substitute answer. Transport failures and truncated responses make a batch incomplete.

The final compact prompt omits the earlier conversation and other observation fields. This makes the current task a test of structured evidence transfer and combination. It does not establish that the model understood every claim in the dialogue.

## The implemented scoring rubric

There is no single weighted “collaboration score.” We report the following measures separately so a model cannot compensate for wrong answers by sending many messages.

| Measure | Exact rule | What it supports |
| --- | --- | --- |
| Answer accuracy | Correct final plan selections divided by all scheduled submitters; missing or invalid reports count as incorrect | Whether residents selected the feasible plan |
| Supported success | Correct plan plus all three evidence IDs, each actually received by that resident | Whether the answer is accompanied by complete available support |
| Team success | At least one resident has a supported success | Whether somebody in the team completed the task; interpret alongside per-resident results |
| Evidence coverage | Distinct known clues divided by three, for each resident | How much necessary information reached each resident |
| New evidence deliveries | Each newly delivered clue-recipient pair; repeat delivery earns no additional count | Observable information transfer, not a claim about internal reasoning |
| Invalid actions | Count of actions that violate the action schema, phase rules or evidence-access rules | Execution reliability |
| Resource use | Scheduled/executed turns, token usage and elapsed time | Cost of the policy; distinguish cached responses from fresh inference |

A legal move or a repeated message may be unhelpful without being invalid. The current grader does not attach a semantic usefulness score to every sentence. Evidence records are transferred by ID rather than reconstructed from model prose, so passing this task does not prove that unrestricted natural-language communication is accurate.

The environment rejects citations or shares of evidence the resident never received. Those are enforcement rules, not evidence that the model independently chose to behave ethically.

## How we isolate the effect of communication

We generate the same underlying case in three conditions:

- **Communication on:** three residents start with one clue each and can share.
- **Communication off:** the same private clues, resident schedule and action budget, but no evidence or message delivery. Peer actions and locations do not provide alternative signaling channels.
- **Single agent with all clues:** one resident starts with complete information. This checks task solvability and uses fewer aggregate actions than the team conditions.

Compare answer accuracy with communication on versus off on matched cases. Full evidence is impossible in the disabled condition, so a supported-success gap alone cannot demonstrate a communication benefit. A resident with one clue can still guess correctly. Resident answers within a case are correlated; they are not independent benchmark cases.

The all-clues condition is not a compute-matched comparison. In the latest small batch, its three final requests exactly matched earlier requests and reused their cached responses. They are consistency checks, not three additional independent samples. The separate single-agent diagnostic has 60 fresh responses over 30 cases and two inference replicates.

## What “collaboration ethics” means here

There is **no implemented or validated ethics score**. The current task authorizes clue sharing and uses synthetic information. It does not test whether agents respect consent, distribute resources fairly, resist incentives to deceive or handle sensitive information appropriately.

Future ethics scenarios would need explicit situations and reviewed criteria. For example, a privacy scenario would specify which facts may be shared with which residents; a fairness scenario would need a justified allocation rule and explicit treatment of competing interests. We should retain disagreement about those criteria instead of inventing an objective ethics score from message counts or success rates.

Human-review packets are prepared but unrated. Each asks a narrow question about whether an action contributed relevant evidence or sought missing information, with `yes`, `no`, `not_applicable` and `insufficient_context` options. Packets distinguish the environment observation from the exact prompt sent to the model. No expert review, human calibration or model-judge validation has occurred.

## Current evidence and scope

The local model is Qwen3-8B in 4-bit, served by vLLM Metal. No persona prompts, activation steering, training updates, paid inference or expert hours are used.

The compact single-agent final-turn diagnostic achieved **60/60 valid, correct, supported decisions** on 30 new instances with two inference replicates. The next three-case collaboration check produced **9/9 correct resident answers with communication** and **1/9 without communication**, with no invalid actions. It comprises three cases, not nine independent cases. These results establish a working engineering baseline for this one task family, not general social competence or a reliable model leaderboard.

See [the result report and complete traces](local-v1-results.md), [technical task design and prior work](distributed-clues-evaluation.md), and [local setup](howto-local-collaboration.md).
