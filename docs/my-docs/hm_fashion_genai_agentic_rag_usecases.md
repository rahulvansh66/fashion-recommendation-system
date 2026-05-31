# H&M Personalized Fashion Recommendations Dataset: GenAI + Agentic RAG Use Cases and Assistant Capabilities

## 1. Purpose

This document maps the **H&M Personalized Fashion Recommendations** Kaggle dataset into practical real-world capabilities for a **GenAI assistant using agentic RAG** over multimodal data: **text, images, structured product metadata, customer attributes, prices, transactions, and time-series behavior**.

The original Kaggle task is to recommend products a customer may purchase in the next 7-day period using historical transactions plus customer and product metadata. The dataset is especially useful because it combines:

- Product catalog metadata
- Customer metadata
- Purchase history
- Product descriptions
- Product images
- Numeric attributes such as age, price, sales channel, and time

This makes it a strong foundation for building a retail/fashion AI assistant that can answer questions, search visually and semantically, recommend outfits, support merchandisers, and power agentic workflows.

---

## 2. Dataset Components

### 2.1 `articles.csv` — Product / Article Metadata

Each row represents one product article available for purchase.

Typical fields:

| Field | Type | Practical Meaning | GenAI / RAG Usage |
|---|---:|---|---|
| `article_id` | ID | Unique product/article identifier | Primary key for product retrieval, image lookup, recommendations |
| `product_code` | ID | Product family/code | Group variants of same product family |
| `prod_name` | Text | Product name | Search, product title generation, query matching |
| `product_type_no` | Numeric/category | Encoded product type | Filters, product taxonomy, candidate generation |
| `product_type_name` | Text/category | Human-readable product type | Natural language retrieval, explanations |
| `product_group_name` | Text/category | Broader product group | Category-level recommendations |
| `graphical_appearance_no` | Numeric/category | Pattern code | Structured filter, image-text consistency check |
| `graphical_appearance_name` | Text/category | Pattern, e.g. solid, striped, printed | Style search and outfit matching |
| `colour_group_code` | Numeric/category | Color code | Color filtering and analytics |
| `colour_group_name` | Text/category | Color name | Natural language color search |
| `perceived_colour_value_id` | Numeric/category | Perceived light/dark value | Visual style matching, contrast matching |
| `perceived_colour_value_name` | Text/category | Dark, light, dusty light, etc. | Fashion explanation and pairing |
| `perceived_colour_master_id` | Numeric/category | Master color group ID | Color-family retrieval |
| `perceived_colour_master_name` | Text/category | Master color name | Color taxonomy and prompts |
| `department_no` | Numeric/category | Department code | Merchandising and department analytics |
| `department_name` | Text/category | Department name | Department-specific assistant answers |
| `index_code` | Category | Catalog index code | High-level catalog filtering |
| `index_name` | Text/category | Index name | Customer-facing product grouping |
| `index_group_no` | Numeric/category | Index group code | Segmentation |
| `index_group_name` | Text/category | Ladieswear, Menswear, Kids, etc. | Gender/age/style group retrieval |
| `section_no` | Numeric/category | Section code | Section-level merchandising |
| `section_name` | Text/category | Section name | More granular category browsing |
| `garment_group_no` | Numeric/category | Garment group code | Fashion taxonomy |
| `garment_group_name` | Text/category | Jersey basic, accessories, etc. | Outfit logic and product grouping |
| `detail_desc` | Text | Product description | Embeddings, semantic search, product Q&A, LLM explanations |

### 2.2 `customers.csv` — Customer Metadata

Each row represents one anonymized customer.

Typical fields:

| Field | Type | Practical Meaning | GenAI / RAG Usage |
|---|---:|---|---|
| `customer_id` | ID | Unique anonymized customer | User profile key |
| `FN` | Binary / nullable | Fashion news flag | Communication preference / propensity feature |
| `Active` | Binary / nullable | Active communication/customer status | Engagement and churn modeling |
| `club_member_status` | Category | Membership state | Loyalty segmentation |
| `fashion_news_frequency` | Category | Fashion-news subscription frequency | Marketing personalization |
| `age` | Numeric | Customer age | Age-aware recommendations and cohort analytics |
| `postal_code` | Hashed/string | Anonymized postal identifier | Geo/cohort grouping without direct location exposure |

### 2.3 `transactions_train.csv` — Purchase History

Each row is a purchase event. Duplicate rows can represent repeated purchases of the same article.

Typical fields:

| Field | Type | Practical Meaning | GenAI / RAG Usage |
|---|---:|---|---|
| `t_dat` | Date | Transaction date | Recency, seasonality, trend analysis |
| `customer_id` | ID | Purchasing customer | Join to profile and purchase history |
| `article_id` | ID | Purchased product | Join to product metadata and image |
| `price` | Numeric | Purchase price | Price sensitivity, basket value, discount logic |
| `sales_channel_id` | Category/numeric | Channel 1 or 2 | Online/offline behavior, channel personalization |

### 2.4 `images/` — Product Images

The dataset includes article images organized by article ID prefixes. Images are critical for multimodal use cases:

- Visual product search
- Similar-looking item retrieval
- Outfit compatibility
- Color/pattern validation
- Product image captioning
- Style classification
- Image-to-product matching

### 2.5 `sample_submission.csv`

This contains the expected recommendation output format:

| Field | Meaning |
|---|---|
| `customer_id` | Customer to recommend for |
| `prediction` | Space-separated list of article IDs |

In a production assistant, this maps to “give me the top N recommendations for this customer.”

---

## 3. Dataset Value: What Can Be Learned

### 3.1 Product Understanding

The assistant can learn:

- Product type, group, department, section, garment group
- Color and perceived color value
- Pattern and graphical appearance
- Textual product descriptions
- Similarity between products from metadata, descriptions, and images
- Product variants grouped by product code
- Product gaps in catalog coverage

### 3.2 Customer Understanding

The assistant can infer:

- Customer purchase preferences
- Price sensitivity
- Preferred colors, garment groups, departments, product types
- Recent intent from latest purchases
- Long-term style profile from historical purchases
- Channel preference
- Loyalty / club status signals
- Age cohort preferences
- Dormant or churn-risk customers

### 3.3 Transactional and Behavioral Understanding

The assistant can derive:

- Recency, frequency, monetary value
- Repeat purchase behavior
- Seasonal buying patterns
- Trending products
- Cohort-level preferences
- Product co-purchase relationships
- Basket composition
- Product lifecycle and demand decay
- Channel-level demand differences

### 3.4 Multimodal Understanding

By combining metadata, text, image, and transaction data, the assistant can support:

- “Find visually similar items”
- “Find black oversized tops under a preferred price range”
- “Recommend items that match this skirt”
- “Explain why this item is recommended”
- “Show alternatives in similar color but different garment group”
- “Detect mismatch between image color and catalog color label”

---

## 4. Recommended Multimodal Agentic RAG Architecture

## 4.1 Data Layers

### Structured Store

Use a relational database or lakehouse table for:

- `articles`
- `customers`
- `transactions`
- recommendation outputs
- feature tables
- aggregate metrics

Best for:

- Exact filters
- Joins
- numeric calculations
- time windows
- cohort analysis
- SQL-based business questions

### Vector Store

Use a vector database for embeddings:

| Embedding Type | Source | Use |
|---|---|---|
| Text embedding | `prod_name`, `detail_desc`, category fields | Semantic product search |
| Image embedding | Product images | Visual similarity search |
| Multimodal embedding | Combined image + text | Cross-modal retrieval |
| Customer profile embedding | Aggregated purchase/style profile | Customer-to-product matching |
| Product behavior embedding | Co-purchase and sequence signals | Collaborative retrieval |

### Feature Store

Store reusable features:

- Customer RFM features
- product popularity by time window
- product age / lifecycle
- customer preferred colors
- customer preferred garment groups
- price sensitivity score
- channel preference
- recent purchase embeddings
- seasonality indicators

### Model Store

Models may include:

- Candidate retrieval models
- ranking models
- image embedding model
- text embedding model
- LLM for explanation and conversation
- forecasting models
- segmentation models
- churn models

---

## 4.2 Agent Tools

An agentic assistant should have tools like:

| Tool | What It Does |
|---|---|
| `search_products_by_text` | Semantic product search using description/category embeddings |
| `search_products_by_image` | Visual similarity search using image embeddings |
| `filter_catalog` | SQL filters by category, color, price, section, garment group |
| `get_customer_profile` | Pulls structured customer profile and derived preferences |
| `get_purchase_history` | Retrieves recent and historical transactions |
| `recommend_for_customer` | Returns ranked personalized recommendations |
| `explain_recommendation` | Produces customer-friendly reasoning |
| `compare_products` | Compares products by metadata, image similarity, price, category |
| `build_outfit` | Selects complementary products |
| `forecast_demand` | Predicts sales or next-week product demand |
| `detect_trend` | Finds trending categories, colors, products |
| `generate_campaign` | Creates personalized marketing copy based on retrieved context |
| `audit_bias_privacy` | Checks age/cohort targeting, sensitive-use constraints, leakage risks |

---

## 4.3 Retrieval Strategy

A strong assistant should not use only one retrieval method. It should use hybrid retrieval:

1. **Structured retrieval**: exact filters, date windows, customer features, price bands.
2. **Text semantic retrieval**: product descriptions and names.
3. **Image retrieval**: visual similarity from product photos.
4. **Collaborative retrieval**: customers who bought similar items also bought.
5. **Time-aware retrieval**: recent popularity, seasonality, freshness.
6. **Business-rule retrieval**: stock, margins, campaign priorities, safety constraints.
7. **LLM synthesis**: explain, compare, summarize, and personalize.

Recommended ranking pipeline:

```text
User query / customer context
        ↓
Intent detection
        ↓
Candidate generation from multiple retrievers
        ↓
Candidate merge + deduplication
        ↓
Feature enrichment
        ↓
Ranker / re-ranker
        ↓
Policy and business guardrails
        ↓
LLM explanation and response
```

---

# 5. Assistant Capabilities and Use Cases

## 5.1 Basic Use Cases

### 1. Product Catalog Q&A

Example questions:

- “What product types are available?”
- “Show me black dresses.”
- “Find items in Ladieswear with solid pattern.”
- “What does this article ID represent?”

Capabilities:

- Join article metadata with images
- Convert catalog fields into natural-language answers
- Explain product taxonomy
- Filter products by color, department, product group, section, garment group

### 2. Product Detail Assistant

Example:

- “Describe article 0706016001 in customer-friendly language.”
- “Generate a short ecommerce product description.”
- “Convert this product metadata into SEO copy.”

Capabilities:

- Summarize product metadata
- Rewrite product descriptions
- Generate titles, tags, bullet points
- Create multilingual descriptions

### 3. Text-Based Product Search

Example:

- “Find a light beige basic top for summer.”
- “I want something similar to a black jersey tank.”

Capabilities:

- Embed `prod_name`, `detail_desc`, and taxonomy fields
- Retrieve semantically similar products
- Apply filters like color, garment group, age segment, price

### 4. Basic Personalized Recommendations

Example:

- “Recommend 12 items for this customer.”

Capabilities:

- Use recent purchases
- Match preferred colors/product groups
- Include popular items from similar cohorts
- Exclude already purchased items if needed

### 5. Simple Recommendation Explanation

Example:

- “Why did you recommend these products?”

Capabilities:

- Explain using purchase history, color preference, category preference, and popularity
- Produce human-friendly explanations instead of raw model scores

---

## 5.2 Intermediate Use Cases

### 6. Multimodal Product Search

Example:

- “Find products visually similar to this image.”
- “Find a similar item but in black.”
- “Find tops that look like this but are more casual.”

Capabilities:

- Use image embeddings
- Combine visual search with metadata filters
- Use LLM to translate visual attributes into query constraints
- Return visually similar products with explanations

### 7. Outfit Builder

Example:

- “Create an outfit around this skirt.”
- “Build a casual weekend look for a 25-year-old customer who likes black basics.”

Capabilities:

- Use garment-group compatibility rules
- Combine product metadata, color theory, image similarity, and customer preferences
- Retrieve complementary items across categories
- Explain outfit logic

### 8. Product Comparison Assistant

Example:

- “Compare these two articles.”
- “Which is better for a customer who usually buys dark basics?”

Capabilities:

- Compare category, color, product group, price, popularity, and image similarity
- Explain differences in user-friendly language
- Recommend one based on customer style profile

### 9. Price Sensitivity Personalization

Example:

- “Recommend items within this customer’s usual price range.”

Capabilities:

- Estimate customer price range from historical purchases
- Avoid recommending products too far outside normal behavior
- Use price bands in ranking

### 10. Channel-Aware Recommendations

Example:

- “Recommend products this customer is likely to buy online.”

Capabilities:

- Use `sales_channel_id` to distinguish channel behavior
- Rank products differently by channel
- Identify items with high channel-specific conversion potential

### 11. Cohort-Based Recommendations

Example:

- “What do similar customers aged 25–35 buy after purchasing black trousers?”

Capabilities:

- Segment customers by age, membership, channel, product preference
- Retrieve co-purchase and next-purchase patterns
- Generate cohort explanations

### 12. Trend Discovery Assistant

Example:

- “What colors are trending this month?”
- “Which garment groups gained momentum recently?”

Capabilities:

- Use transaction time-series
- Calculate growth over rolling windows
- Summarize trends by category/color/department
- Generate merchandising insights

### 13. Product Substitution Assistant

Example:

- “Find alternatives to this article.”

Capabilities:

- Retrieve similar products using metadata, text, image, and customer behavior
- Match same product type but different color/price
- Match same style but different garment group
- Explain substitutions

---

## 5.3 Advanced Use Cases

### 14. Agentic Personal Shopper

Example:

- “I need clothes for a business-casual trip next week. I like dark colors and usually buy basics.”

Agent workflow:

1. Detect occasion, constraints, and preferences.
2. Retrieve customer purchase history.
3. Infer style profile.
4. Retrieve candidate tops, bottoms, outerwear, accessories.
5. Use image/text embeddings for style coherence.
6. Rank by customer preference, price sensitivity, and popularity.
7. Generate outfit sets.
8. Explain each outfit.

Capabilities:

- Conversational shopping
- Multi-step planning
- Product retrieval
- Visual matching
- Personalized explanation
- Constraint satisfaction

### 15. Next-Best-Action Marketing Agent

Example:

- “For this customer, should we send a discount, style recommendation, or reactivation campaign?”

Capabilities:

- Churn-risk estimation
- Purchase propensity estimation
- Campaign type selection
- Personalized copy generation
- Recommended product bundle generation
- Frequency capping and communication preference checks

### 16. Customer Style Memory

Example:

- “Summarize this customer’s style.”

Capabilities:

- Build a natural-language customer profile:
  - preferred colors
  - preferred garment groups
  - average spend
  - recent style shift
  - channel preference
  - likely next category
- Useful for CRM, chatbots, and stylist tools

### 17. Multimodal Recommendation Explanation

Example:

- “Explain recommendations using both product image and purchase history.”

Capabilities:

- Explain visual similarity:
  - “similar dark tone”
  - “same casual basic style”
  - “same garment family”
- Explain behavioral match:
  - “you recently bought similar tops”
  - “this matches your preferred price range”

### 18. Demand Forecasting Assistant

Example:

- “Which products are likely to sell next week?”

Capabilities:

- Use transaction dates, prices, categories, and popularity trends
- Forecast item-level or category-level demand
- Explain drivers of forecast
- Flag demand spikes and drops

### 19. Inventory and Merchandising Copilot

Example:

- “Which product groups should we promote for young customers this week?”

Capabilities:

- Cohort demand analysis
- Category trend analysis
- Color trend analysis
- Product lifecycle analysis
- Campaign candidate generation

### 20. Product Lifecycle Intelligence

Example:

- “Which items are declining in demand?”

Capabilities:

- Detect demand decay
- Identify stale products
- Recommend markdown candidates
- Recommend products to refresh or bundle

### 21. Visual Catalog Quality Agent

Example:

- “Find products where image color seems inconsistent with catalog color.”

Capabilities:

- Use vision model to detect dominant colors
- Compare detected image features to metadata fields
- Flag potential catalog errors
- Improve product data quality

### 22. Cold-Start Recommender

Example:

- “Recommend products to a new customer with only age and a stated preference.”

Capabilities:

- Use product metadata and popularity
- Use age/cohort patterns
- Use natural-language stated preferences
- Use visual and text similarity instead of transaction history

### 23. Graph-Based Fashion Recommendation

Example:

- “Find items related through customer-product-product relationships.”

Capabilities:

- Model graph nodes: customers, articles, product types, colors, departments
- Model edges: purchased, co-purchased, same color, same group, visually similar
- Use graph retrieval or graph neural networks for recommendations

### 24. Generative Product Bundling

Example:

- “Create bundles for customers who buy jersey basics.”

Capabilities:

- Discover frequent co-purchases
- Generate bundles by outfit logic
- Optimize for price bands and diversity
- Produce bundle names and marketing copy

### 25. Natural-Language Business Analyst

Example:

- “What happened to black garment upper-body sales over the last 30 days?”

Capabilities:

- Translate question into SQL/time-series analysis
- Join transactions and articles
- Produce summarized insight
- Generate chart-ready tables
- Explain confidence and caveats

### 26. Personalized Search Re-Ranking

Example:

- Customer searches “dress”; assistant re-ranks results for that customer.

Capabilities:

- Retrieve broad product candidates
- Re-rank using customer profile
- Balance relevance, novelty, price, trendiness, and diversity

### 27. Conversational Preference Refinement

Example:

- User: “Show me dresses.”
- Assistant: “Do you prefer dark colors, casual styles, or partywear?”

Capabilities:

- Ask targeted clarifying questions
- Update temporary session profile
- Re-rank results after each answer
- Preserve explainability

### 28. Returns Reduction Assistant

The dataset does not contain returns directly, but the assistant can approximate risk factors using proxies such as unusual category, unusual price band, or mismatch with customer’s history.

Capabilities:

- Flag low-confidence recommendations
- Prefer familiar sizes/styles if size data is available externally
- Explain why a recommendation may be risky
- Suggest safer alternatives

### 29. Sustainability-Oriented Recommendations

Example:

- “Recommend versatile items instead of one-off trend items.”

Capabilities:

- Prefer basics, neutral colors, repeatable garment groups
- Recommend items compatible with prior purchases
- Reduce low-fit recommendations
- Support lower-return shopping behavior

### 30. Synthetic Styling Data Generation

Example:

- “Generate training examples of outfit explanations.”

Capabilities:

- Use product metadata and images to generate synthetic descriptions
- Create question-answer pairs for product search
- Create instruction-tuning data for a fashion assistant

---

# 6. Practical Assistant Modes

## 6.1 Customer-Facing Assistant

Primary users: shoppers.

Capabilities:

- Personalized recommendations
- Visual product search
- Outfit generation
- Product comparison
- “Complete the look”
- Budget-aware recommendations
- Trend-aware suggestions
- Explanation of why items match

Example response:

> “Based on your recent purchases of black jersey basics and your usual price range, I recommend these dark neutral tops and trousers. I included one lighter option for variety, but kept the silhouette close to what you usually buy.”

## 6.2 Stylist Assistant

Primary users: human stylists or store associates.

Capabilities:

- Customer style summary
- Outfit board generation
- Alternatives and substitutions
- Occasion-based outfit planning
- Visual similarity search
- Customer conversation support

## 6.3 Merchandising Assistant

Primary users: merchandisers, planners, buyers.

Capabilities:

- Trend discovery
- product/category performance summaries
- demand forecasting
- cohort-level product preferences
- underperforming product detection
- campaign product selection

## 6.4 Marketing Assistant

Primary users: CRM, lifecycle marketing, campaign teams.

Capabilities:

- Segment generation
- personalized campaign copy
- product recommendations per segment
- churn-risk campaigns
- next-best-action suggestions
- fashion-news targeting

## 6.5 Data Quality Assistant

Primary users: catalog operations and data teams.

Capabilities:

- Missing or weak product description detection
- image-metadata mismatch detection
- duplicate / near-duplicate product detection
- taxonomy inconsistency detection
- invalid or unusual price detection

## 6.6 Data Science Copilot

Primary users: ML engineers and data scientists.

Capabilities:

- Feature generation suggestions
- experiment design
- retrieval/ranking pipeline explanation
- model evaluation support
- error analysis
- recommendation debugging

---

# 7. Example Agentic RAG Workflows

## 7.1 “Recommend Products for Customer” Workflow

```text
Input: customer_id
1. Fetch customer metadata.
2. Fetch recent transactions.
3. Join transactions with article metadata.
4. Build customer style profile.
5. Retrieve candidates using:
   - similar product metadata
   - similar product images
   - co-purchase patterns
   - trending items
6. Remove already purchased items if required.
7. Re-rank by relevance, recency, price fit, diversity, and business rules.
8. Generate natural-language explanation.
Output: ranked article list + explanations
```

## 7.2 “Search by Uploaded Image” Workflow

```text
Input: user image
1. Generate image embedding.
2. Retrieve visually similar product images.
3. Enrich results with article metadata.
4. Apply user filters, e.g. color, product type, price.
5. Re-rank with text-image similarity and popularity.
6. Explain visual matches.
Output: similar products + why they match
```

## 7.3 “Business Question Answering” Workflow

```text
Input: natural-language analytics question
1. Detect entities: dates, colors, categories, age groups, channels.
2. Translate to SQL or dataframe query.
3. Join transactions with articles/customers.
4. Compute metrics.
5. Retrieve supporting context.
6. Generate concise business explanation.
Output: answer + table/chart-ready data + caveats
```

## 7.4 “Build Outfit” Workflow

```text
Input: seed article_id or natural-language occasion
1. Retrieve seed product metadata and image.
2. Determine outfit slots: top, bottom, outerwear, shoes/accessories if available.
3. Retrieve compatible products by category, color, style, and image similarity.
4. Personalize using customer history.
5. Ensure diversity and avoid duplicates.
6. Explain outfit composition.
Output: outfit bundle + rationale
```

---

# 8. Feature Engineering Ideas

## 8.1 Customer Features

- Days since last purchase
- Number of purchases in last 7 / 30 / 90 days
- Average purchase price
- Median purchase price
- Preferred product groups
- Preferred colors
- Preferred departments
- Preferred sales channel
- Purchase diversity score
- Repeat purchase rate
- Recent style shift score
- Customer embedding from purchased article embeddings

## 8.2 Product Features

- Product popularity by week
- Product popularity by age cohort
- Product popularity by channel
- Product lifecycle stage
- Product image embedding
- Product text embedding
- Product metadata embedding
- Co-purchase embedding
- Similar products by image/text/behavior

## 8.3 Transaction Features

- Product recency trend
- Seasonal trend
- Price trend
- Basket composition
- Co-purchase matrix
- Sequential next-item patterns
- Channel-specific conversion proxies

## 8.4 Multimodal Features

- Dominant image color
- Image-text consistency score
- Visual similarity clusters
- Style cluster
- Product image caption
- Color contrast score for outfit building
- Visual novelty vs customer history

---

# 9. Evaluation Metrics

## 9.1 Recommendation Metrics

- MAP@12
- Recall@K
- Precision@K
- NDCG@K
- Hit rate
- Coverage
- Diversity
- Novelty
- Serendipity
- Repeat purchase lift

## 9.2 Search Metrics

- Query relevance
- Click-through rate
- Add-to-cart rate
- Conversion rate
- Filter satisfaction rate
- Zero-result rate

## 9.3 Assistant Quality Metrics

- Groundedness
- Explanation accuracy
- Hallucination rate
- User satisfaction
- Clarification success rate
- Task completion rate
- Latency

## 9.4 Business Metrics

- Revenue per session
- Conversion uplift
- Average order value
- Retention uplift
- Churn reduction
- Return reduction proxy
- Campaign engagement

---

# 10. Guardrails and Practical Constraints

## 10.1 Privacy

The dataset contains anonymized customer IDs and hashed postal codes, but production systems must still enforce:

- No re-identification attempts
- Minimal use of personal attributes
- Privacy-safe segmentation
- Access control for customer data
- Clear separation between analytics and customer-facing responses

## 10.2 Fairness

Age and membership data can be useful but should be handled carefully:

- Avoid discriminatory targeting
- Avoid stereotypes in explanations
- Do not infer sensitive traits
- Use age only where business-appropriate and policy-compliant

## 10.3 Hallucination Control

The assistant should:

- Retrieve product data before describing a product
- Cite or expose source product fields internally
- Avoid inventing unavailable attributes such as size, material, stock, or availability unless integrated from external systems
- Clearly say when data is missing

## 10.4 Dataset Limitations

Important limitations:

- No explicit user ratings
- No product inventory availability
- No returns data
- No product size fit feedback
- No live stock or current catalog status
- Postal codes are anonymized
- Images may not exist for every article
- Historical data may not represent current trends

---

# 11. Implementation Blueprint

## 11.1 Minimal MVP

Build:

- Product catalog search
- Customer purchase-history lookup
- Popularity-based recommendations
- Simple personalized recommendations
- Product explanation generation

Recommended stack:

- SQL database for structured data
- Vector database for product text embeddings
- LLM for explanation
- Batch feature pipeline

## 11.2 Strong V1

Add:

- Image embeddings
- Hybrid retrieval
- Customer embeddings
- Candidate generation + ranker
- Visual similarity search
- Trend assistant
- Business analytics Q&A

## 11.3 Advanced Production Version

Add:

- Agent planner
- Tool calling
- graph retrieval
- real-time session behavior
- inventory and stock integration
- pricing and promotion rules
- A/B testing
- monitoring and drift detection
- human feedback loop

---

# 12. Example Assistant Questions Supported by This Dataset

## Shopper Questions

- “What should I buy next?”
- “Find something similar to this product.”
- “Show me black basics I might like.”
- “Build an outfit from my recent purchases.”
- “Why are you recommending this?”
- “Show me alternatives in lighter colors.”

## Stylist Questions

- “Summarize this customer’s style.”
- “Find three outfits for this customer.”
- “What products pair well with this article?”
- “What is a safer alternative for this customer?”

## Merchandising Questions

- “Which colors are trending among young customers?”
- “Which product groups are declining?”
- “Which items should be promoted this week?”
- “Which products are often bought together?”

## Data Science Questions

- “What features should I build for ranking?”
- “Why did the recommender miss this purchase?”
- “Which customers are cold-start?”
- “Which products have weak metadata?”

---

# 13. High-Value Use Case Prioritization

| Priority | Use Case | Why It Matters |
|---:|---|---|
| 1 | Personalized recommendations | Core dataset objective and direct business value |
| 2 | Hybrid product search | Uses text + metadata + image together |
| 3 | Recommendation explanation | Makes AI outputs trustworthy |
| 4 | Outfit builder | Strong customer-facing GenAI differentiator |
| 5 | Trend and merchandising assistant | High business-user value |
| 6 | Visual similarity search | Strong multimodal capability |
| 7 | Next-best-action marketing | Extends recommender into CRM |
| 8 | Data quality agent | Improves catalog quality and retrieval quality |
| 9 | Demand forecasting | Useful for planning and inventory |
| 10 | Graph-based recommendations | Advanced retrieval and ranking improvement |

---

# 14. Summary

This dataset can support far more than a Kaggle recommender model. With GenAI, multimodal embeddings, structured retrieval, and agentic workflows, it can power a complete fashion retail assistant ecosystem:

- Customer-facing personal shopper
- Visual search assistant
- Stylist copilot
- Merchandising analyst
- Marketing personalization agent
- Catalog quality auditor
- Data science copilot

The strongest architecture is a **hybrid agentic RAG system** that combines:

- SQL retrieval over customers, articles, and transactions
- Text embeddings over product descriptions and metadata
- Image embeddings over product photos
- Behavioral embeddings over purchase history
- Ranking models for recommendation quality
- LLMs for explanation, conversation, and workflow orchestration

The most practical first implementation is a recommendation and product-search assistant. The most advanced implementation is a multimodal personal-shopping agent that can understand customer history, product images, product descriptions, prices, trends, and business rules at the same time.

---

# 15. References

- Kaggle competition data page: H&M Personalized Fashion Recommendations — product recommendations from previous purchases; includes `articles.csv`, `customers.csv`, `transactions_train.csv`, `sample_submission.csv`, and images.
- Kaggle competition overview: states that metadata spans garment type, customer age, product descriptions, and garment images.
- RelBench `rel-hm`: reports a relational version with 3 tables, 16,664,809 rows, 37 columns, a 7-day time window, and tasks such as user-item purchase prediction, item-sales prediction, transaction-price prediction, and user churn prediction.
