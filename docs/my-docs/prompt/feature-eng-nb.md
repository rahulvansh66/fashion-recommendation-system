refer @docs/implementation-info/guides/features-eng.md 

dataset schema info: @docs/system-design/schema-info.md

Refer : @docs/system-design/v1/v1-requirements.md @docs/system-design/v1/v1-hld.md 

create feature-engineering jupyter notebook in @notebooks/, which later will be useful to create feature enginerring data pipeline

Include following in notebook:
1. Move data to s3 () or 
- local : Move `dataset\dummy` to `s3\dataset\dummy` and move  `dataset\sample_2000_users` to `s3\dataset\sample_2000_users`
- aws : Upload data to s3 bucket equivalent to above location 

2. Split dataset in train, val using cutoff date 2020-03-31 and next 7 days after cutoff for testdata. e.g.
Train data : From start to 2020-03-24
Val data : From 2020-03-25 to 2020-03-31
Test data : From 2020-04-01 to 2020-04-07

store splited data in s3 dir based on local/aws config

3. add code for feature engieering on train data

Consider following things while writing notebook:

- notebook should be in proper structred section/subsections.
- Done write too much of code in one cell, divide into multiple cells
- add makrdown text to add descreption related code block
- code should have required comments for easy undestanding, in function, add docstring that includes functin desc, Parameters and returns
- use pysypark, and code should be such that it's easy to migrate to local to aws with minimal changes