- generate python arch diagram and review with chatgpt and claude

Please review the file at docs/system-design/hld.md and let me know if you want to make any changes before we move on to writing out the implementation plan. Once you approve, the next step would be invoking the writing-plans skill to break this HLD into a sequenced implementation plan.
- use aws reference to implemet two tower


things to taker care while plan
- high cost things like load balencer should be implemented at last, so it wont be call all time
- indipendent modul should be mentioned, as all is cursor generated, so we want to build indipendent moduals paraleelly.... like... code generation of datapreprocessing steps, ml training, evaluation modules can be done in parallel, and can be tested with unit testing with respective dummy data, later we can do intigration testing. make graph for my visulization.
- first implement main functionality like end to end model pipleine, then implemeted secondnd level functionalities like cach, queue etc...
- use aws reference to implemet two tower

-------
- things like should be implicit feedback should be stored/track will increase cost, analyse cost difference with and w/o
- lld before implementation plan

## Relationship Model

```
CUSTOMERS (1) ─────< (M) TRANSACTIONS >(M)─── (1) ARTICLES
  - customer_id         - customer_id            - article_id
  - demographics        - article_id      
  - preferences         - price, date, channel
```
each customer can make multiple transactions, and each transaction can involve multiple articles.


