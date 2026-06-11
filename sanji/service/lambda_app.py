"""Zappa/Lambda entrypoint: builds the app at import (cold start).

sandjig verifies/creates its DynamoDB tables during construction — in deployed
environments the tables are pre-provisioned by the sandjig resources stack
(``sandjig template --resources-only``), so this reduces to a DescribeTable.
"""

from sanji.service.app import create_app

app = create_app()
