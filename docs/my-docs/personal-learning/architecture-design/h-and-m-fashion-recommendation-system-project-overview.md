# Building an H&M Fashion Recommendation System

Most recommendation projects look simple from the outside. A user opens an app, sees a few products they might like, clicks one, and moves on. Under the hood, though, a good recommender is a small distributed system: data pipelines, model training, vector search, ranking, caching, fallbacks, monitoring, and a product surface that makes the result feel instant.

This project is my version of that system, built around the H&M Personalization Challenge dataset. The goal is not just to train a model in a notebook. The goal is to build a practical, end-to-end fashion recommendation platform that feels close to how a real production recommender would be designed, while still staying small enough to run, understand, and tear down without burning money.

The v1 system serves personalized top-10 article recommendations through a simple web experience. A user logs in, chooses one of six active customers from the dataset, and sees recommended fashion items update in the page. Behind that small interaction is the full serving path: cache, retrieve, filter, rank, and order.

## Why This Project Exists

Fashion recommendation is a good problem because it has all the interesting parts of recommendation systems without needing a massive product surface. Customers have purchase histories. Articles have categories, prices, colors, and product metadata. Some items are popular, some are niche, and people often want a mix of familiar and slightly different choices.

That makes the project useful for learning three things at once:

- How to turn raw retail data into model-ready features.
- How to combine retrieval and ranking instead of asking one model to do everything.
- How to serve recommendations with real engineering concerns like latency, cost, caching, and failure handling.

The system is intentionally scoped as a learning-grade, production-pattern v1. It is not pretending to be a full H&M-scale production system on day one. But the architecture is designed so the same patterns can scale from a small development sample to the full dataset without changing the basic shape of the system.

## The User Experience

The web app is deliberately simple. It uses FastAPI, Jinja2, HTMX, and Tailwind CSS, all running in one ECS Fargate service. There is no separate frontend build pipeline and no SPA framework. The user gets a server-rendered page, clicks a customer card, and HTMX swaps in the recommendations without a full page refresh.

For v1, authentication is intentionally minimal: the demo uses the placeholder `rr / rr` login. That is enough for a portfolio-style project, and the production path is documented separately as Cognito plus a JWT authorizer.

The interesting part of the UI is the latency story. The page shows six active users. The first three are pre-warmed overnight, so their recommendations should come straight from Redis in roughly 15 ms. The last three run the full recommendation pipeline on demand, closer to 190 ms when everything is warm.

That difference is visible, and that is the point. It turns caching from an invisible backend optimization into something you can actually demonstrate.

## The Recommendation Flow

The online path follows a five-stage pipeline:

1. Check the result cache.
2. Retrieve candidate items.
3. Filter out items the user has already purchased.
4. Rank the remaining candidates.
5. Reorder the final list for a little more variety.

The cache check comes first because the cheapest and fastest recommendation is the one already computed. If Redis has a fresh `reco:{customer_id}` entry, the app returns it immediately and skips the downstream ML calls.

On a cache miss, the system fetches user features, sends them to a two-tower user model hosted on SageMaker, and receives a 256-dimensional user embedding. That embedding is sent to a FAISS Lambda, which searches the item vector index and returns the top candidate article IDs.

Those candidates are not final recommendations yet. The pipeline removes anything the customer has already purchased, enriches the remaining candidates with user, item, and cross features, and sends the batch to a XGBoost ranker. XGBoost scores the candidates by predicted purchase probability.

The last step is a diversity-aware reorder. The top four positions stay focused on raw relevance. Positions five and six are allowed to introduce some variety based on product type, color group, and price bucket. The final positions return to the next strongest ranked items. It is a small rule, but it reflects a real product intuition: a recommendation list should not be ten near-duplicates.

## Why Use Two Models?

The system separates retrieval from ranking because those jobs are different.

The two-tower model is good at finding a broad set of likely items quickly. It maps users and items into the same embedding space, so FAISS can search for similar item vectors without scoring every article one by one.

XGBoost then does the slower, more precise work on a much smaller candidate set. It can use richer features, including cross features like category match, price affinity, and recency signals. This two-stage shape is common in real recommender systems because it balances speed and quality.

Trying to rank every item directly would be wasteful. Relying only on nearest-neighbor retrieval would leave quality on the table. The split gives each model a job it is suited for.

## The Data Backbone

S3 is the system of record. Raw H&M CSV files land in the `raw/` zone, cleaned parquet data is written to `clean/`, features go to `features/`, and model artifacts, embeddings, and FAISS indices live in their own zones.

Redis is deliberately treated as a cache, not a source of truth. It holds hot user features, item features, seen-item sets, popular fallback lists, active users for the UI, and computed recommendation results. If Redis is empty, the data can be rebuilt from S3 and the offline pipelines.

That distinction matters. It keeps the architecture simpler and avoids turning the cache into a hidden database.

## Offline Pipelines

The offline side prepares everything the online path needs to stay fast.

AWS Glue handles data preparation and feature engineering. It reads the raw H&M files, validates and deduplicates records, writes parquet outputs, builds user and item features, and refreshes Redis keys such as popular items, seen-item sets, and the six active users shown in the app.

SageMaker Pipelines handles model training. The flow builds training tables, trains the two-tower model and XGBoost ranker, evaluates both models, registers approved versions, computes item embeddings, builds a FAISS index, and rolls new models out through canary deployment.

This keeps the request path lightweight. The online service should not be doing heavy feature engineering or model-building work while a user is waiting.

## Cache Pre-Warming

One of the most practical parts of the design is the pre-warming workflow.

Every day, an EventBridge rule triggers a small producer Lambda. It reads the top three active users from Redis and sends one message per customer to an SQS queue. A consumer Lambda processes those messages, runs the same recommendation pipeline used by the live API path, and writes the results back to Redis with a 12-hour TTL.

The consumer is idempotent using a `SETNX prewarm:done:{customer_id}:{date}` guard, so duplicate SQS delivery does not corrupt the cache or repeatedly invoke SageMaker. Failed messages go to a DLQ after retries, and CloudWatch alarms make failures visible.

This is a nice example of a production pattern in a small project: asynchronous work queue, idempotent consumer, retry safety, and a direct user-facing benefit.

## Running on AWS Without Overbuilding

The v1 architecture is AWS-native, but cost-conscious.

The app runs as a unified FastAPI service on ECS Fargate. API Gateway HTTP API fronts it through VPC Link and Cloud Map, avoiding the monthly idle cost of an ALB for this low-traffic use case. FAISS runs in Lambda and loads the index from S3. Redis uses ElastiCache. Training and inference use SageMaker where the ML-specific operational features are valuable.

The biggest cost is SageMaker endpoints, so the design assumes they are destroyed between active learning sessions. Terraform is part of the core architecture for that reason: the whole stack should be deployable with `terraform apply` and removable with `terraform destroy`.

The project is not trying to be the cheapest possible toy demo. It is trying to spend money only where it teaches a real production pattern.

## Reliability and Fallbacks

The serving path is designed to return something useful even when a dependency is unhealthy.

Every downstream call is wrapped with a circuit breaker. If the user-tower endpoint fails, the system can use a cached user embedding or fall back to popular items. If FAISS fails, it can return popular items by category. If XGBoost fails, it can use the FAISS candidates ordered by similarity and still apply the diversity step. If multiple ML dependencies are unavailable, the response can degrade to popular items with a `degraded` flag.

That means failure is visible, but it does not have to become a blank page.

## What V1 Does Not Try To Do

The project is intentionally scoped. V1 does not include real-time event ingestion, online learning, Cognito, RAG, LLM-based tagging, or a microservices split. Those are useful future directions, but adding them now would make the first version harder to finish and harder to explain.

The guiding idea is simple: build the smallest version that still demonstrates the real system shape.

## What Makes This Project Interesting

What I like about this design is that the architecture has a clear story.

The user sees a small recommendation page. The system behind it shows the real mechanics: a cache-first request path, a two-stage ML recommender, vector search, ranking, diversity, batch features, model deployment, queue-based pre-warming, observability, and infrastructure that can be created and destroyed on demand.

It is not just a model. It is the surrounding system that makes the model usable.

That is the main lesson of the project: recommendation quality matters, but production recommendation systems are won or lost in the glue around the model. Data freshness, latency, fallback behavior, deployment safety, and cost control are part of the product too.

## References

- [V1 requirements](../v1-requirements.md)
- [V1 high-level design](../v1-hld.md)
