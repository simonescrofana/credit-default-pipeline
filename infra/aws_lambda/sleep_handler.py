"""Sleep-on-inactivity Lambda handler.

This module checks how long ago the last request was seen and, if the
inactivity threshold has been exceeded, stops the RDS instance and
scales the ECS services back down to zero, keeping the deployed project
idle (and cost-free) between visits.

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
INACTIVITY_THRESHOLD_SECONDS = int(os.environ["INACTIVITY_THRESHOLD_SECONDS"])

table = dynamodb.Table(TABLE_NAME)
STATE_KEY = "state"


def handler(event: dict, context: object) -> dict:
    """Suspend the project's infrastructure after enough inactivity.

    Reads the current status and last-activity timestamp from DynamoDB.
    Does nothing if the system is already asleep, or if it is awake but
    the inactivity threshold has not been reached yet. Otherwise, stops
    the RDS instance, scales the ECS services down to zero, updates the
    stored status to "asleep", and publishes a notification via SNS.

    Args:
        event (dict): The EventBridge scheduled trigger payload. Unused:
            this handler only reacts to the current stored state, not to
            anything in the trigger itself.
        context (object): The Lambda runtime context. Unused.

    Returns:
        `dict`: A `statusCode` and a short `body` describing whether the
            system was put to sleep, already asleep, or still active.

    """
    item = table.get_item(Key={"id": STATE_KEY}).get("Item")

    if item is None or item.get("status") != "awake":
        return {"statusCode": 200, "body": "already asleep"}

    now = int(time.time())
    idle_for = now - int(item.get("last_request_at", now))

    if idle_for < INACTIVITY_THRESHOLD_SECONDS:
        return {"statusCode": 200, "body": "still active"}

    rds.stop_db_instance(DBInstanceIdentifier=DB_INSTANCE_ID)
    ecs.update_service(cluster=CLUSTER_NAME, service=API_SERVICE_NAME, desiredCount=0)
    ecs.update_service(cluster=CLUSTER_NAME, service=UI_SERVICE_NAME, desiredCount=0)

    table.put_item(
        Item={
            "id": STATE_KEY,
            "status": "asleep",
            "last_request_at": item.get("last_request_at", now),
        }
    )

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Project going to sleep",
        Message=(
            f"No activity for over {INACTIVITY_THRESHOLD_SECONDS // 60} minutes - "
            "RDS and ECS are shutting down."
        ),
    )

    return {"statusCode": 200, "body": "going to sleep"}
