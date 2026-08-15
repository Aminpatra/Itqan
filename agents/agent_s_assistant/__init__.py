"""Agent S — the assistant over a user's own results.

The sixth agent, and the first that is not a producer. A through E read
documents or the web and write envelopes; Agent S reads what they wrote and
answers questions about it, for one signed-in user, about that user only.

It is also the first place in this system where a sentence somebody types can
cause work to happen, and the design follows from that:

**The model cannot act.** It returns `answer` or `propose_rerun` as a structured
field. A rerun happens only when the user makes a separate, explicit HTTP
request that says so. The model suggests, the person decides, the code executes
— the same shape the pipeline already uses when it pauses at
`awaiting_confirmation` and waits for a human.

**The model cannot reach another user's data**, and not because it is told not
to. It is handed a fact sheet built in code from one user's mapped results;
there is nothing else in the prompt to leak. Every store method behind it is
scoped by a `user_id` that comes from the signed session cookie and is never a
parameter a request can supply.

**The model cannot spend quota.** Limits are claimed by a guarded UPDATE in
`AppStore.claim_quota` before this graph is built at all. Over quota, nothing
here runs.

That division is deliberate and it is the lesson this repo has paid for three
times: `work_arrangement` was fabricated on 19 of 19 postings against a prompt
forbidding it, Agent E called an unpriced course free once in 25 draws with the
warning present, and Agent A's model paraphrased quotes it was told to copy.
Instructions are not controls. The controls are in the code around this package.
"""
