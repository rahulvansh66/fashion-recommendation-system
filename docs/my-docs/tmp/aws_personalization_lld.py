"""
AWS Personalization Architecture - LLD Diagram
Tool: Python 'diagrams' library  (https://diagrams.mingrammer.com)
Run on: Replit (replit.com) or locally

Install dependencies:
    pip install diagrams
    # Also requires Graphviz: https://graphviz.org/download/

Usage:
    python aws_personalization_lld.py
    # Generates: aws_personalization_architecture.png
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.integration import APIGateway
from diagrams.aws.analytics import KinesisDataStreams, KinesisDataFirehose
from diagrams.aws.storage import S3
from diagrams.aws.compute import Lambda
from diagrams.aws.ml import Personalize
from diagrams.aws.general import General
from diagrams.onprem.client import Client

graph_attrs = {
    "fontsize": "13",
    "bgcolor": "white",
    "pad": "0.5",
    "splines": "ortho",
}

with Diagram(
    "AWS Real-Time Personalization Architecture",
    show=True,
    filename="aws_personalization_architecture",
    direction="LR",
    graph_attr=graph_attrs,
):
    # ── Client ──────────────────────────────────────────────────────────────
    client = Client("Browser / App\n(Click Stream)")

    # ── Ingestion Pipeline (top row) ─────────────────────────────────────
    api_gw_ingest = APIGateway("API Gateway\n(Ingest)")
    kinesis_streams = KinesisDataStreams("Kinesis\nData Streams")
    kinesis_firehose = KinesisDataFirehose("Kinesis\nData Firehose")
    s3_bucket = S3("S3\n(Interactions)")

    # ── Stream Consumer Lambda ───────────────────────────────────────────
    lambda_stream = Lambda("Lambda\n(Stream Consumer)")

    # ── Amazon Personalize cluster ───────────────────────────────────────
    with Cluster("Amazon Personalize"):

        with Cluster("① Dataset Group"):
            item_meta   = General("Item\nMetadata")
            user_meta   = General("User\nMetadata")
            interactions = General("Interactions")

        with Cluster("② Model Training"):
            solution_ver = General("Solution\nVersion")
            campaign     = General("Campaign")

        event_tracker = General("Event\nTracker")

    # ── Recommendations path (bottom row) ───────────────────────────────
    api_gw_recs  = APIGateway("API Gateway\n(Recommendations)")
    lambda_recs  = Lambda("Lambda\n(getRecommendations)")

    # ════════════════════════════════════════════════════════════════════
    # Connections
    # ════════════════════════════════════════════════════════════════════

    # Click stream → ingestion pipeline
    client >> Edge(label="Click Stream\nEvents") >> api_gw_ingest
    api_gw_ingest >> Edge(label="Put Events") >> kinesis_streams
    kinesis_streams >> kinesis_firehose
    kinesis_firehose >> Edge(label="Persist\nInteractions") >> s3_bucket

    # Kinesis → Lambda → Personalize Event Tracker  (step 3)
    kinesis_streams >> Edge(label="③ Consume Stream") >> lambda_stream
    lambda_stream >> Edge(label="Put Events") >> event_tracker

    # Personalize internals  (step 2 → step 1 → step 2 flow)
    event_tracker >> interactions
    [item_meta, user_meta, interactions] >> solution_ver
    solution_ver >> campaign

    # Recommendations API path  (step 3)
    client >> Edge(label="Recommendations\nAPI call") >> api_gw_recs
    api_gw_recs >> Edge(label="③ getRecommendations") >> lambda_recs
    lambda_recs >> campaign
