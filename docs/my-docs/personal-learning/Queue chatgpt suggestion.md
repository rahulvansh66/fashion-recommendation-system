# Fashion Recommendation System HLD Review – Conversation Notes

## Topics Covered

### 1. HLD Review Summary
- Architecture is generally good for a learning/demo-grade production-style recommender.
- Main concerns:
  - "Always-on Lambda demo" conflicts with destroying SageMaker/Redis resources.
  - Too many serving entry points (Fargate + Lambda + multiple API Gateways).
  - Redis is overloaded with many responsibilities.
  - S3 fallback in online serving path may hurt latency.
  - FAISS-in-Lambda has cold-start and scaling limitations.
  - Authentication is demo-only (`rr/rr`).
  - 12-hour cache may lead to stale recommendations.
  - Seen-set rebuilt nightly can miss same-day interactions.

### 2. Do We Need SQS/Kinesis in v1?
**Answer: No, not in the synchronous recommendation path.**

Recommendation requests are synchronous:

Cache → Retrieve → Filter → Rank → Order → Return Result

Adding queues would:
- Increase complexity
- Add latency
- Not improve user experience

Queues become useful when:
- Processing click events
- Processing purchases
- Cache invalidation
- Feature updates
- Event ingestion

---

## Queue Concepts Explained

### Async Feature Updates
Instead of updating user features during the request:

Without SQS:

User Click
→ Update Redis
→ Update DB
→ Update Features
→ Return

With SQS:

User Click
→ Put message in SQS
→ Return immediately
→ Background worker updates features

Useful once live user events exist.

---

### Fanout (SNS + SQS)

One event needs to trigger multiple actions.

Example:

Purchase Event

May need:
- Analytics
- Cache invalidation
- Feature update
- Email
- Reporting

SNS publishes once.

Multiple queues receive copies:

Purchase
→ SNS
   → Analytics Queue
   → Feature Queue
   → Cache Queue

Each consumer works independently.

---

### DLQ (Dead Letter Queue)

Stores messages that repeatedly fail processing.

Example:

Queue
→ Lambda
→ Fails 3 times
→ Message moved to DLQ

Useful for:
- Debugging
- Replay
- Failure investigation

---

## Option A – Batch Failure Handling

Original idea:
- DLQ for Glue/SageMaker failures

Refined recommendation:
- Use Step Functions Retry/Catch
- Use SNS notifications
- Optional SQS failure queue

Recommended flow:

EventBridge
→ Step Functions
→ Retry failures
→ Catch permanent failures
→ SNS alert
→ Optional SQS failure queue

---

## Option B – Cache Pre-Warming (Recommended)

Strongest queue-related addition.

Problem:

Recommendation cache expires overnight.

Morning:

User
→ Cache miss
→ Full pipeline runs

Solution:

Nightly Pipeline
→ Identify active users
→ Push customer_ids to SQS

Consumers:

SQS
→ Lambda Workers
→ Generate recommendations
→ Write to Redis

Benefits:
- Better latency
- Fewer morning cache misses
- Controls concurrency
- Strong interview talking point

---

## Option C – SNS Fanout for Ops Events

Technically valid but likely over-engineering for v1.

Example:

CloudWatch Alarm
→ SNS
   → Pager Queue
   → Digest Queue
   → Audit Queue

Recommendation:
- Keep CloudWatch + SNS email only.
- Defer full fanout architecture.

---

## Step Functions Explained

Step Functions = Workflow Orchestrator

Example:

1. Clean Data
2. Build Features
3. Train Model
4. Build FAISS Index
5. Populate Redis

Benefits:
- Visual workflow
- Retry support
- Failure handling
- Monitoring

---

### Retry

If a step fails:

Run
→ Fail
→ Retry
→ Retry
→ Succeed

Useful for transient cloud failures.

---

### Catch

If retries fail:

Run
→ Retry
→ Retry
→ Still fails
→ Catch block executes

Catch can:
- Send alerts
- Record failure
- Trigger compensating actions

---

### SNS

Notification broadcaster.

Pipeline Failure
→ SNS Topic
→ Email
→ Lambda
→ Queue

Simple use:

Pipeline Failure
→ SNS
→ Email to owner

---

### SQS Failure Queue

Stores failure details.

Example payload:

```json
{
  "pipeline": "nightly-feature-pipeline",
  "failed_step": "GlueFeatureEngineering",
  "error": "S3 timeout"
}
```

Useful for:
- Auditing
- Replay
- Troubleshooting

---

## Seen Set Discussion

### What is a Seen Set?

Stores items a user has already interacted with.

Example:

seen:123 = {
 item_10,
 item_25,
 item_87
}

---

### How It Is Used

Retriever returns:

[item_10, item_25, item_40]

Filter removes seen items:

[item_40]

---

### Problem with Nightly Rebuild

2 AM:

seen:123 = {
 item_10,
 item_25
}

10 AM:

User clicks item_40

Redis still contains:

seen:123 = {
 item_10,
 item_25
}

System may recommend item_40 again.

---

### Why This Happens

v1 has no:

- POST /events
- Event ingestion
- Live feature updates

Therefore seen sets are based on historical data only.

---

### Future Fix (v1.1)

User Click
→ POST /events
→ SQS
→ Lambda
→ Redis Seen Set Update

Redis:

SADD seen:123 item_40

Now item_40 is filtered immediately.

---

## Final Recommendations

### Add
1. Step Functions Retry/Catch
2. SNS failure notifications
3. SQS cache pre-warming queue (best queue-related enhancement)

### Optional
1. SQS failure queue for pipeline failures

### Do Not Add Yet
1. SNS→SQS ops fanout architecture
2. Event-driven real-time pipelines (move to v1.1)

### Strong Interview Story

"I intentionally avoided queues in the synchronous recommendation path because recommendation serving requires immediate responses. I used SQS only for asynchronous workloads such as cache pre-warming. After the nightly feature pipeline completes, active users are queued, Lambda workers precompute recommendations, and Redis is warmed before traffic arrives."
