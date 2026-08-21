"""Tests for the sleep Lambda handler.

Covers the relevant happy paths: the system is already asleep (no-op),
the system is awake but still within the inactivity threshold (no-op),
and the system is awake past the threshold (stop RDS/ECS, notify). boto3
clients are mocked - no real AWS calls.

"""

import importlib
import os
import time
from unittest.mock import MagicMock, patch

import pytest

ENV = {
    "DYNAMODB_TABLE": "test-wake-state",
    "DB_INSTANCE_IDENTIFIER": "test-db",
    "ECS_CLUSTER_NAME": "test-cluster",
    "ECS_API_SERVICE_NAME": "test-api",
    "ECS_UI_SERVICE_NAME": "test-ui",
    "SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789012:test-topic",
    "INACTIVITY_THRESHOLD_SECONDS": "1800",
}


@pytest.fixture
def handler_module():
    """Import the sleep handler fresh, with boto3 and the environment mocked.

    See the equivalent fixture in `test_handler.py` (wake Lambda tests)
    for why `patch` is used as a context manager rather than a decorator
    here.

    Yields:
        `tuple[ModuleType, MagicMock, MagicMock, MagicMock, MagicMock]`:
            The handler module, mocked DynamoDB table, mocked RDS client,
            mocked ECS client, and mocked SNS client.

    """
    with (
        patch.dict(os.environ, ENV),
        patch("boto3.resource") as mock_resource,
        patch("boto3.client") as mock_client,
    ):
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table

        mock_rds = MagicMock()
        mock_ecs = MagicMock()
        mock_sns = MagicMock()

        def client_side_effect(service_name, *args, **kwargs):
            return {"rds": mock_rds, "ecs": mock_ecs, "sns": mock_sns}[service_name]

        mock_client.side_effect = client_side_effect

        import infra.aws_lambda.sleep_handler as handler_mod

        importlib.reload(handler_mod)

        yield handler_mod, mock_table, mock_rds, mock_ecs, mock_sns


def test_already_asleep_is_a_noop(handler_module):
    """The handler should not touch RDS/ECS/SNS if already asleep."""
    handler_mod, mock_table, mock_rds, mock_ecs, mock_sns = handler_module
    mock_table.get_item.return_value = {"Item": {"id": "state", "status": "asleep"}}

    result = handler_mod.handler({}, None)

    assert result["statusCode"] == 200
    assert "already asleep" in result["body"]
    mock_rds.stop_db_instance.assert_not_called()
    mock_ecs.update_service.assert_not_called()
    mock_sns.publish.assert_not_called()


def test_awake_but_still_active_is_a_noop(handler_module):
    """The handler should not sleep if the inactivity threshold is unmet."""
    handler_mod, mock_table, mock_rds, mock_ecs, mock_sns = handler_module
    recent = int(time.time()) - 60
    mock_table.get_item.return_value = {
        "Item": {"id": "state", "status": "awake", "last_request_at": recent}
    }

    result = handler_mod.handler({}, None)

    assert result["statusCode"] == 200
    assert "still active" in result["body"]
    mock_rds.stop_db_instance.assert_not_called()
    mock_ecs.update_service.assert_not_called()
    mock_sns.publish.assert_not_called()


def test_awake_past_threshold_goes_to_sleep(handler_module):
    """The handler should stop RDS, scale down ECS, and notify via SNS."""
    handler_mod, mock_table, mock_rds, mock_ecs, mock_sns = handler_module
    stale = int(time.time()) - 3600
    mock_table.get_item.return_value = {
        "Item": {"id": "state", "status": "awake", "last_request_at": stale}
    }

    result = handler_mod.handler({}, None)

    assert result["statusCode"] == 200
    assert "going to sleep" in result["body"]

    mock_rds.stop_db_instance.assert_called_once_with(DBInstanceIdentifier="test-db")
    assert mock_ecs.update_service.call_count == 2
    for call in mock_ecs.update_service.call_args_list:
        assert call.kwargs["desiredCount"] == 0
    mock_sns.publish.assert_called_once()

    last_put_item = mock_table.put_item.call_args.kwargs["Item"]
    assert last_put_item["status"] == "asleep"
