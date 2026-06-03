# A Fashion Recommendation System, From Ten Thousand Feet

I wanted to build a recommender the way a real company would build one — and then run it on the budget of someone who pays for their own AWS account. That tension, between "do it properly" and "don't go broke," is the whole story of this project. Almost every decision you'll read about below is really an answer to the same quiet question: *what's the cheapest honest version of the grown-up thing?*

This is a walk through the system at a comfortable altitude. I'm not going to hand you line counts or config files here. I want to explain what each part is *for*, and why it earns its place. If you've ever wondered how "show this person ten things they'll like" turns into an actual running system, this is the shape of it.

---

## The Goal, Stated Plainly

Take a customer, hand back ten fashion items worth their attention. Do it quickly, do it cheaply, and don't let the whole thing fall apart when one component has a bad day.

There's a second goal underneath the obvious one, and it matters more than it sounds. This is a learning system wearing a production system's clothes. It's designed for a scale it will never see — millions of users — and then deployed against a tiny slice of data. The point isn't to serve real traffic. The point is that you can stand in front of any piece of it and say, with a straight face, "here's why a serious team would do it this way." Every box has to justify itself.

---

## A Philosophy Before an Architecture

Before any of the components made sense, a handful of ground rules did the deciding for me.

Spend money only where it teaches something or genuinely can't be avoided. Default to things that sleep when nobody's using them. Make the version that runs on my laptop and the version that runs in the cloud the *same* thing, separated only by a few environment values — never by different logic. And treat every component as a polite stranger to its neighbors: it should be swappable without anyone next to it noticing.

None of that is glamorous, but it's the part that holds. The architecture is downstream of these rules.

---

## Turning a Question Into an Answer

The core of the system is a short, deliberate sequence — a kind of assembly line that a customer's identity walks down and comes out the other end as a ranked list.

The lesson hiding in this layer is that good recommendations aren't a single clever model; they're a series of small, honest steps. First, check whether we've already figured this out recently. If not, narrow an entire catalog down to a promising handful. Drop the things they've already bought. Carefully score what remains. Then gently rearrange the finalists so the list isn't ten variations of the same shirt.

What I like most about it is the restraint. The genuinely heavy thinking is done in advance, away from the request. By the time someone is actually waiting on a screen, the work left to do is light. Nobody should have to wait while a webpage makes up its mind about their taste.

---

## Memory, and the Trick It Enables

If the pipeline is the engine, the cache is the system's short-term memory — and, conveniently, its best party trick.

The idea behind caching is almost too obvious to state: don't redo expensive work you've already done. But this system goes out of its way to make that idea *visible*. Some users have their answers quietly worked out ahead of time, overnight. Others are figured out live, the moment you ask. Click the first kind and the result is just *there*. Click the second and you catch the system thinking for a moment. Same machinery, completely different feel — which turns out to be the most honest way I've found to explain why caching is worth the trouble, no whiteboard required.

That overnight preparation is also a tidy little example of how real systems quietly hand work off to themselves in the background, with the guardrails that stop them from accidentally doing the same job twice.

---

## The Expensive, Clever Bits

The actual machine learning lives off to one side: the parts that learn what a user is like, find items that resemble what they want, and judge how likely a purchase really is.

I kept this corner walled off for two reasons, one principled and one purely about money. These are the smart and costly pieces, so I treat them like specialists you bring in for a job rather than staff you keep on the clock all day. Walling them off also means they can be retrained, replaced, or improved entirely on their own schedule, without the rest of the system ever needing to care that anything changed.

---

## Somewhere to Keep Everything

None of the cleverness means anything without a dependable place to keep the ingredients — customer histories, item details, the precomputed features, the trained models themselves.

This layer's only ambition is to be the boring, trustworthy source of truth: organized, durable, and inexpensive. There's a firm rule baked into it that I'm fond of — there's one real home for data, and the fast layer that sits in front of it is *only* a convenience. The fast layer is allowed to forget. The real home is not. Lose the convenience and nothing of value is actually gone; it just has to warm back up.

---

## The Work That Happens at Night

While nobody is looking, a set of scheduled jobs does the unglamorous heavy lifting: tidying raw data, computing fresh features, retraining the models, and stocking the shelves for the next day.

This is the kitchen prep before the restaurant opens. The daytime system gets to feel effortless precisely *because* something tedious ran at three in the morning. By the time the first real request shows up, the catalog's been combed through and everything it needs is already waiting.

---

## Watching Itself

A surprising slice of the system exists purely to keep an eye on the rest of it.

This is the part you don't think about until something feels wrong. Is it slow? Is a piece failing quietly without saying so? Is the cache genuinely helping, or just pretending to? Just as importantly, the system is built so that when something *does* break, it sags instead of shattering — a slightly worse answer still beats an error page every single time. That's the whole difference between a system that's "down" and one that's merely "having a rough moment."

---

## Locks Worth Fitting

Even a project handling pretend data deserves to be built as though it weren't.

So the habits here are the grown-up ones: encrypt things by default, hand out only the permissions actually needed, and never leave secrets sitting in the open. The way users sign in is, admittedly, a stand-in — and I've labeled it honestly as exactly that — because the goal was to get every *other* lock right, not to reinvent the login screen for the thousandth time.

---

## The Machine That Ships the Machine

Finally, there's the plumbing that carries a code change all the way to something actually running, safely.

The aim here is to make change *boring*, which is the highest compliment you can pay a deployment process. A change gets tested, built, and rolled out, with deliberate pauses before anything risky and a quick way back if a new model starts behaving badly. And the part I'm quietly proudest of: the entire system can be conjured into existence with a single command and dismissed just as easily with another. That's not just neatness — it's the most important cost-control feature in the design. When the session ends, the meter stops.

---

## So, Why Go This Far?

You could throw together a recommendation demo in an afternoon with a notebook and a spreadsheet. This is deliberately not that. This is the version that takes the scenic route, because the scenic route is where you actually learn how caching, queues, fallbacks, safe deployments, and one-command infrastructure fit together once they have to share a system instead of a tutorial.

The reward is that the whole thing is explainable, end to end. Every component has a reason. Every tradeoff has a story behind it. And when I'm done for the day, I can watch a cached answer outrun a fresh one in real time, tear the entire thing down, and go to sleep knowing tomorrow's bill is zero.

For a side project, that feels like the right kind of ending.
