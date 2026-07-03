"""
Generate the V1 Fashion Recommendation System HLD architecture diagram (v3)
using AWS service icons via the `diagrams` Python library.

v3 changes vs v1:
  - Left-to-right flow (user on the left) for easier reading.
  - Simplified edge/ingress (no ACM TLS cert, VPC Link, or Cloud Map).
  - Observability & Security collapsed to a tiny block (IAM + CloudWatch +
    SNS) with no cross-cutting dotted lines, to reduce visual clutter.

Prerequisites:
  brew install graphviz         # macOS
  sudo apt-get install graphviz # Ubuntu/Debian
  pip install diagrams

Run:
  python create_hld_architecture_diagram_v3.py

Output:
  fashion_reco_v1_hld_architecture_v3.png
  fashion_reco_v1_hld_architecture_v3.svg
"""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.blank import Blank
from diagrams.onprem.client import Client, User


def aws_icon(module: str, *class_names: str):
    """Return the first available diagrams AWS icon class, else Blank.

    The diagrams package occasionally renames or omits less common AWS icons
    between releases. This helper keeps the script runnable while still using
    real AWS icons wherever the installed diagrams version supports them.
    """
    try:
        mod = import_module(module)
        for class_name in class_names:
            icon = getattr(mod, class_name, None)
            if icon is not None:
                return icon
    except Exception:
        pass
    return Blank


# Network / edge
CloudFront = aws_icon("diagrams.aws.network", "CloudFront")
APIGateway = aws_icon("diagrams.aws.network", "APIGateway", "ApiGateway")

# Compute / app
Fargate = aws_icon("diagrams.aws.compute", "Fargate")
Lambda = aws_icon("diagrams.aws.compute", "Lambda")
ECR = aws_icon("diagrams.aws.devtools", "ECR")

# Data / ML
S3 = aws_icon("diagrams.aws.storage", "S3")
ElastiCache = aws_icon("diagrams.aws.database", "ElastiCache")
SageMaker = aws_icon("diagrams.aws.ml", "Sagemaker", "SageMaker")
SageMakerTraining = aws_icon("diagrams.aws.ml", "SagemakerTrainingJob", "SageMakerTrainingJob", "Sagemaker")
SageMakerModel = aws_icon("diagrams.aws.ml", "SagemakerModel", "SageMakerModel", "Sagemaker")

# Integration / orchestration / observability
EventBridge = aws_icon("diagrams.aws.integration", "Eventbridge", "EventBridge")
SQS = aws_icon("diagrams.aws.integration", "SQS")
StepFunctions = aws_icon("diagrams.aws.integration", "StepFunctions")
Glue = aws_icon("diagrams.aws.analytics", "Glue")
CloudWatch = aws_icon("diagrams.aws.management", "Cloudwatch", "CloudWatch")
SNS = aws_icon("diagrams.aws.integration", "SNS")
IAM = aws_icon("diagrams.aws.security", "IAM")


def connect_chain(nodes: Iterable, **edge_kwargs) -> None:
    """Connect nodes left-to-right/top-to-bottom as a chain."""
    previous = None
    edge = Edge(**edge_kwargs) if edge_kwargs else None
    for node in nodes:
        if previous is not None:
            if edge is None:
                previous >> node
            else:
                previous >> edge >> node
        previous = node


graph_attr = {
    "fontsize": "22",
    "bgcolor": "white",
    "pad": "0.5",
    "splines": "ortho",
    "nodesep": "0.65",
    "ranksep": "1.1",
}

node_attr = {
    "fontsize": "12",
    "fontname": "Arial",
}

edge_attr = {
    "fontsize": "10",
    "fontname": "Arial",
    "color": "#4a5568",
}

with Diagram(
    "Fashion Recommendation System - V1 HLD Architecture (v3)",
    filename="fashion_reco_v1_hld_architecture_v3",
    outformat=["png", "svg"],
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    # ---- 1) Client (far left) ----
    with Cluster("Client"):
        reviewer = User("End user /\nportfolio reviewer")
        browser = Client("Browser\nJinja + HTMX")

    # ---- 2) Edge + Ingress (simplified) ----
    with Cluster("Edge + Ingress"):
        cloudfront = CloudFront("CloudFront\nTLS + static cache")
        api_gw = APIGateway("API Gateway\nHTTP API\nthrottle 60 RPS")

    # ---- 3) Application Layer ----
    with Cluster("Application Layer"):
        app = Fargate("ECS Fargate task\nFastAPI monolith\nJinja2 + HTMX + API\n0.5 vCPU / 1 GB")
        ecr = ECR("ECR\napp image")

    # ---- 4) Online Serving Pipeline (the hot path, left-to-right) ----
    with Cluster("Online Serving Pipeline"):
        cache_check = ElastiCache("Stage 0\nRedis cache check\n12h TTL")
        retrieve = SageMaker("Stage 1a\nSageMaker\nuser tower")
        faiss = Lambda("Stage 1b\nLambda + FAISS\ntop-100 retrieval")
        filter_seen = ElastiCache("Stage 2\nRedis seen-set filter")
        rank = SageMaker("Stage 3\nSageMaker\nXGBoost ranker")
        order = Lambda("Stage 4\nDiversity reorder")

    # ---- 5) Data Stores ----
    with Cluster("Data Stores"):
        redis = ElastiCache("ElastiCache Redis\nresult cache + features\nseen sets + rate limits")
        s3 = S3("S3 data lake\nraw / clean / features\nmodels / embeddings / indices")

    # ---- 6) Offline Batch + ML Pipelines ----
    with Cluster("Offline Batch + ML Pipelines"):
        weekly_cron = EventBridge("EventBridge\nweekly + daily cron")
        step_fn = StepFunctions("Step Functions\ndata + feature pipeline")
        glue = Glue("AWS Glue\nPySpark jobs")
        sm_pipeline = SageMaker("SageMaker Pipelines\ntrain-register-deploy")
        sm_training = SageMakerTraining("SageMaker Training\ntwo-tower + XGBoost")
        sm_registry = SageMakerModel("SageMaker Model Registry\napproval + versions")
        index_builder = Lambda("Lambda\nbuild FAISS index")

    # ---- 7) Cache Pre-Warmer ----
    with Cluster("Cache Pre-Warmer"):
        warm_cron = EventBridge("EventBridge\ndaily 05:00 UTC")
        producer = Lambda("prewarm-producer")
        queue = SQS("SQS\ncache-prewarm-queue")
        dlq = SQS("SQS DLQ\nmax receive 3")
        consumer = Lambda("prewarm-consumer\nconcurrency 5")

    # ---- 8) Observability & Security (tiny block, no cross-cutting lines) ----
    with Cluster("Observability + Security"):
        iam = IAM("IAM")
        cw = CloudWatch("CloudWatch")
        sns = SNS("SNS alerts")
        cw >> Edge(label="alarms") >> sns

    # ===================== Request / data flow =====================

    # User request path (left -> right)
    reviewer >> browser
    browser >> Edge(label="HTTPS") >> cloudfront >> api_gw >> app
    ecr >> Edge(style="dotted", label="image") >> app

    # Online serving hot path
    app >> Edge(label="GET /recommendations/{customer_id}") >> cache_check
    cache_check >> Edge(label="cache hit") >> app
    cache_check >> Edge(label="cache miss") >> retrieve >> faiss >> filter_seen >> rank >> order >> Edge(label="SETEX reco:{cid}") >> redis
    redis >> Edge(label="return top-10") >> app

    # App direct data dependencies
    app << Edge(label="cache, features, rate limit") >> redis
    app >> Edge(label="fallback feature read") >> s3
    faiss >> Edge(label="load .index from S3") >> s3

    # Offline pipelines
    connect_chain([weekly_cron, step_fn, glue, s3], label="raw -> clean -> features")
    glue >> Edge(label="refresh popular, seen, active users") >> redis
    step_fn >> Edge(label="trigger") >> sm_pipeline
    sm_pipeline >> sm_training >> sm_registry
    sm_registry >> Edge(label="deploy approved models") >> retrieve
    sm_registry >> Edge(label="deploy approved models") >> rank
    sm_pipeline >> Edge(label="item embeddings") >> index_builder >> Edge(label="write new FAISS index") >> s3

    # Cache pre-warming path
    warm_cron >> producer >> Edge(label="LRANGE active:users:top6") >> redis
    producer >> Edge(label="SendMessageBatch") >> queue >> consumer
    queue >> Edge(style="dashed", label="failed 3x") >> dlq
    consumer >> Edge(label="run same pipeline") >> retrieve
    consumer >> Edge(label="SETNX + SETEX reco") >> redis

print("Created fashion_reco_v1_hld_architecture_v3.png and .svg")
