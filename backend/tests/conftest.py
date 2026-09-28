"""Keep every backend test run isolated from local development data."""

import os
import tempfile


os.environ["QIWEN_DATA_DIR"] = tempfile.mkdtemp(prefix="qiwen-test-")
