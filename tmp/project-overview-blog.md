 # Linkedin Post

 


# Building a Fashion Recommendation System That Won't Bankrupt Me

There's a particular kind of project that every engineer eventually attempts: the one where you try to rebuild a slice of Spotify, Pinterest, or Netflix in your own backyard, on your own dime, mostly to prove to yourself that you actually understand how the magic works. This is that project.

The premise is simple. Given a customer from the H&M dataset, show them ten fashion items they might actually want to buy. The execution is where it gets interesting — because doing this *properly*, with the same patterns the big players use, while keeping the monthly AWS bill closer to a nice dinner than a car payment, turns out to be a genuinely fun engineering problem.

This post is a tour of the system at the altitude of a small plane. Close enough to see the shape of things, high enough that you don't get lost in the weeds. Each section answers one question: *why does this part exist?*

---

## What We're Actually Trying to Do

At its heart, the goal is to take "who is this user" and turn it into "here are ten things for them," fast and cheaply, without the whole thing falling over when one piece misbehaves.

But there's a second, quieter goal hiding underneath: this is a *learning-grade* system built with *production-grade* patterns. It's deliberately designed for a scale it will never actually reach, then deployed on a tiny sample of data. The idea is that you should be able to point at any box in the architecture and explain why a real company would build it that way — even if, in my version, that box is handling about forty requests a month and napping the rest of the time.

---

## The Front Door

Every system needs a face, and this one is a web app you can log into and click around.

The purpose of this layer is to make the whole thing *tangible*. Recommendations are abstract until you see them appear on a screen. So there's a login, a page of customer cards, and a click that makes ten products show up. Nothing fancy, no heavyweight frontend framework with a build step that takes longer than the model training — just enough interface to demonstrate that the engine behind it is real and working.

It also doubles as the stage for the system's favorite party trick, which we'll get to.

---

## The Bouncer at the Door

Before any request gets to do real work, something needs to decide whether it's allowed in and how often it can knock.

This layer exists for two unglamorous but essential reasons: keeping the system reachable in a sane way, and making sure nobody — including an overenthusiastic script, or me at 2 AM — can hammer the expensive machinery into a costly frenzy. It's the velvet rope and the clipboard. Boring when it works, very much missed when it doesn't.

---

## The Pipeline: From "Who?" to "Here You Go"

This is the beating heart of the whole thing — the assembly line that turns a customer ID into a ranked list of products.

The purpose here is to show that good recommendations aren't one clever trick; they're a *sequence* of deliberate steps, each doing one job well. Check if we've already done this work recently. Narrow the entire catalog down to a promising shortlist. Throw out the stuff they've already bought. Carefully rank what's left. Then nudge the final order so it isn't ten nearly-identical shirts.

The deeper lesson baked into this layer is restraint: heavy thinking happens ahead of time, offline. The live request path stays lean and quick, because nobody enjoys waiting for a webpage to decide what they like.

---

## The Cache, and the Party Trick

If the pipeline is the heart, the cache is the system's short-term memory — and its single best demo.

The purpose of caching is obvious in theory: don't redo expensive work you've already done. But this system makes that lesson *visible*. Some users have their recommendations quietly precomputed overnight; others get computed live, on the spot. Click one and the answer appears almost instantly. Click another and you watch it think for a beat. Same system, wildly different feel — and a perfect, honest way to explain why caching matters without waving your hands at a whiteboard.

The overnight precomputation also happens to be a clean little showcase of how real systems hand work off to themselves in the background, complete with all the safety nets that keep it from doing the same job twice.

---

## The Brains (Kept on a Leash)

The actual machine learning lives in its own corner: the models that understand users, find similar items, and score how likely a purchase is.

The purpose of isolating this layer is partly intellectual honesty and partly financial survival. These are the smart, expensive parts of the system — so they're treated like specialists you call in when needed, not staff you keep on payroll around the clock. Keeping them separate also means they can be swapped, retrained, and upgraded without anyone else in the system noticing or caring.

---

## The Pantry

None of the cleverness above works without somewhere to keep the ingredients: customer histories, item details, precomputed features, and the model artifacts themselves.

This layer's job is to be the single, dependable source of truth — organized, durable, and cheap. There's a deliberate philosophy here of keeping things simple: one main place for storage, one fast place for things you need *right now*, and a firm rule that the fast place is just a convenience, never the place that actually *remembers*. Lose it and nothing important is gone, just warmed-up.

---

## The Night Shift

While nobody's watching, a set of scheduled jobs quietly does the heavy lifting: cleaning raw data, computing features, training models, and getting everything ready for the next day.

The purpose of this layer is to keep the daytime fast by doing the slow stuff at night. It's the kitchen prep before the restaurant opens. By the time the first request of the day arrives, the catalog has been combed through, the models are fresh, and the popular items are already on the shelf. The live system gets to look effortless precisely because something tedious happened at 3 AM.

---

## The Watchtower

You can't trust what you can't see, so a fair amount of the system is dedicated purely to watching itself.

This layer exists to answer the questions you only ask when something feels off: Is it slow? Is something failing quietly? Is the cache actually helping, or just pretending to? Crucially, it's also designed so that when a piece *does* fail, the whole thing degrades gracefully rather than collapsing — a worse answer is still an answer, and an answer beats an error page every time. It's the difference between a system that's "down" and one that's just "having a rough moment."

---

## The Locks and the Keys

Even a portfolio project handling fake data deserves to be built like it isn't.

The purpose of this layer is to practice good security hygiene as a default, not an afterthought: encrypt things, hand out only the permissions actually needed, and never, ever leave secrets lying around in the code. The authentication here is admittedly a placeholder — and honestly labeled as such — because the point was to get every *other* lock right, not to reinvent the login form for the thousandth time.

---

## The Robot That Ships the Robot

Finally, there's the machinery that takes a code change and safely walks it all the way to running infrastructure.

The purpose here is to make change *boring* — in the best possible way. Push a change, let the automation test it, build it, and roll it out, with careful checks before anything risky happens and an escape hatch if a new model misbehaves. And tying it all together is the ability to summon the entire system into existence with one command, and banish it just as easily with another. That last part isn't just elegance; it's the single most important cost-control feature in the whole design. When the learning session ends, the meter stops.

---

## Why Bother With All This?

You could build a recommendation demo in an afternoon with a notebook and a CSV file. This is not that. This is the version that takes the long way on purpose — because the long way is where you learn how caching, queues, circuit breakers, canary deployments, and infrastructure-as-code actually fit together when they share a system instead of a tutorial.

The whole thing is designed to be explainable end to end. Every box has a reason. Every tradeoff has a story. And at the end of the day, you can spin it up, watch a cache hit beat a cache miss in real time, click "destroy," and go to bed knowing tomorrow's bill is zero.

Which, for a side project, is its own kind of beautiful.
