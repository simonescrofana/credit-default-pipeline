"""Wake-on-request Lambda handler.

This module wakes the project's ECS services and RDS instance when a
request comes in while the system is asleep, and keeps their activity
timestamp fresh in DynamoDB so the companion sleep Lambda knows not to
suspend them yet.

"""

import os
import time

import boto3

dynamodb = boto3.resource("dynamodb")
rds = boto3.client("rds")
ecs = boto3.client("ecs")
sns = boto3.client("sns")

TABLE_NAME = os.environ["DYNAMODB_TABLE"]
DB_INSTANCE_ID = os.environ["DB_INSTANCE_IDENTIFIER"]
CLUSTER_NAME = os.environ["ECS_CLUSTER_NAME"]
API_SERVICE_NAME = os.environ["ECS_API_SERVICE_NAME"]
UI_SERVICE_NAME = os.environ["ECS_UI_SERVICE_NAME"]
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]

table = dynamodb.Table(TABLE_NAME)
STATE_KEY = "state"


def handler(event: dict, context: object) -> dict:
    """Wake the project's infrastructure if it is currently asleep.

    Reads the current status from DynamoDB. If the system is already
    awake, only refreshes the last-activity timestamp and returns early.
    Otherwise, starts the RDS instance and scales the ECS services back
    up, updates the stored status to "awake", and publishes a
    notification via SNS.

    Args:
        event (dict): The Lambda trigger payload. Unused: any invocation
            is treated as a wake signal regardless of its contents.
        context (object): The Lambda runtime context. Unused.

    Returns:
        `dict`: A `statusCode` and a short `body` describing whether the
            system was already awake or has just been woken up.

    """
    now = int(time.time())

    item = table.get_item(Key={"id": STATE_KEY}).get("Item")
    current_status = item.get("status") if item else "asleep"

    table.put_item(
        Item={
            "id": STATE_KEY,
            "status": current_status,
            "last_request_at": now,
        }
    )

    if current_status == "awake":
        return {"statusCode": 200, "body": "already awake"}

    rds.start_db_instance(DBInstanceIdentifier=DB_INSTANCE_ID)
    ecs.update_service(cluster=CLUSTER_NAME, service=API_SERVICE_NAME, desiredCount=1)
    ecs.update_service(cluster=CLUSTER_NAME, service=UI_SERVICE_NAME, desiredCount=1)

    table.put_item(
        Item={
            "id": STATE_KEY,
            "status": "awake",
            "last_request_at": now,
        }
    )

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Project woken up",
        Message="Someone hit the project URL - RDS and ECS are starting back up.",
    )

    return {"statusCode": 200, "body": "waking up"}
