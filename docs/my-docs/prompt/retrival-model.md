/brainstorming 
- lets implement retrival model in new notebook in @notebooks, you can take reference docs\implementation-info\guides\two-tower-retrieval-training-guide.md.

- you can take data from @s3\dataset\sample_2000_users\features\transactions

- use same input parameteres, model archetecture, hyperparameters, train and evaluation Strategy, positive and negative sample Strategy.  

- train val test split should be as mentioned in @docs\system-design\v1\v1-requirements.md

- add **Popularity correction** like its done in  @tmp\recsys-v2, as fixes a bias from **in-batch negatives**.

- use ml flow for experiment tracking and mangement and optuna for hyper Parameter tuning. 

- use feature to store??

- use best practicies for versoning model experiments, artifacts, dataset. 

-  i just rememberd that use AWS Managed MLflow for experiment tracking and mangement and optuna for hyper Parameter tuning. refer @docs/implementation-info/guides/mlflow-optuna-experiment-guide.md 

- AWS credentials should be maintained from env file. and maintain terraform for creating aws resources as mentiond in@docs/system-design/project-structure.md @docs/system-design/v1/v1-hld.md 


- notebook should be orgainized with section and subsctions, add comments in code, docstring in functions, use markdown before each block 


ask me follow up question if you have doubt or Confusions.

