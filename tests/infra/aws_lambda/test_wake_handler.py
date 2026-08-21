"""Tests for the wake Lambda handler.

Covers the two happy paths only: the system is already awake (refresh
timestamp, do nothing else) and the system is asleep (start RDS/ECS,
notify). boto3 clients are mocked - no real AWS calls.

"""

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

ENV = {
    "DYNAMODB_TABLE": "test-wake-state",
    "DB_INSTANCE_IDENTIFIER": "test-db",
    "ECS_CLUSTER_NAME": "test-cluster",
    "ECS_API_SERVICE_NAME": "test-api",
    "ECS_UI_SERVICE_NAME": "test-ui",
    "SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789012:test-topic",
}


@pytest.fixture
def handler_module():
    """Import the wake handler fresh, with boto3 and the environment mocked.

    The module reads os.environ and calls boto3.resource/client at import
    time, so both the environment and the boto3 mocks must be in place
    before the import happens. `patch` is used as a context manager
    rather than a decorator here: applying `patch.dict` and `patch` as
    stacked decorators on a fixture does not guarantee the environment is
    set before boto3 resolves its region internally, and the import
    fails with `NoRegionError`. The nested `with` block enforces that
    ordering explicitly.

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

        import infra.aws_lambda.wake_handler as handler_mod

        importlib.reload(handler_mod)

        yield handler_mod, mock_table, mock_rds, mock_ecs, mock_sns


def test_already_awake_only_refreshes_timestamp(handler_module):
    """The handler should not touch RDS/ECS/SNS if already awake."""
    handler_mod, mock_table, mock_rds, mock_ecs, mock_sns = handler_module
    mock_table.get_item.return_value = {"Item": {"id": "state", "status": "awake"}}

    result = handler_mod.handler({}, None)

    assert result["statusCode"] == 200
    assert "already awake" in result["body"]
    mock_table.put_item.assert_called_once()
    put_item = mock_table.put_item.call_args.kwargs["Item"]
    assert put_item["status"] == "awake"
    mock_rds.start_db_instance.assert_not_called()
    mock_ecs.update_service.assert_not_called()
    mock_sns.publish.assert_not_called()


def test_asleep_wakes_everything_up(handler_module):
    """The handler should start RDS, scale up ECS, and notify via SNS."""
    handler_mod, mock_table, mock_rds, mock_ecs, mock_sns = handler_module
    mock_table.get_item.return_value = {"Item": {"id": "state", "status": "asleep"}}

    result = handler_mod.handler({}, None)

    assert result["statusCode"] == 200
    assert "waking up" in result["body"]

    mock_rds.start_db_instance.assert_called_once_with(DBInstanceIdentifier="test-db")
    assert mock_ecs.update_service.call_count == 2
    mock_sns.publish.assert_called_once()

    last_put_item = mock_table.put_item.call_args.kwargs["Item"]
    assert last_put_item["status"] == "awake"
