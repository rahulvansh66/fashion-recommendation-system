/brainstorming 
i want to create implemetation plan of computing features step, you can refer to jupytertebook 
@docs/reference-code/notebooks/1_fp_computing_features.ipynb   and its supportive files are attached @docs/reference-code/recsys/features @docs/reference-code/recsys/config.py @docs/reference-code/recsys/raw_data_sources  to under refrence implementation of computing features.

Refer schema info @docs/system-design/schema-info.md 
Refer project structure @system-design/project-structure.md , we have to modularise code this way.
Refer info @docs/system-design/v1/v1-infrastructure-layer.md , if you have different opinion you can ask me. 

you goad is to create detailed implementation document (in .md) for computing features step from the reference files. Also adapt implementation based on our hld design @docs/system-design/v1/v1-hld.md , as this will be a pyspark glue job, which first run and tested locally and then migrate to aws, and as mentioned in hld code should be migration friendly from local to aws with minimal code changes. 


If you have any doubts, ambiguities ask follow-up questions before proceeding.
