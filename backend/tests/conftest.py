"""Keep every backend test run isolated from local development data."""

import os
import tempfile


os.environ["QIWEN_DATA_DIR"] = tempfile.mkdtemp(prefix="qiwen-test-")
# A locally enabled paid provider must never make ordinary tests call its API.
# Live integration evaluation runs in separate explicit CLI profiles.
os.environ["QIWEN_MODEL_MODE"] = "mock"
os.environ["QIWEN_TOOL_PROTOCOL"] = "auto"
