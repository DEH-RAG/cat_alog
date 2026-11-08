"""pytest configuration of the tests of the plugin.

The core has no default for the secrets of its encryption (``CAT_CRYPTO_KEY``, ``CAT_CRYPTO_SALT``): it refuses to start
without them, and loading a plugin reads them. The tests run outside the container, so they get test-only values when
the environment has none.
"""
import os

os.environ.setdefault("CAT_CRYPTO_KEY", "test-only-crypto-key")
os.environ.setdefault("CAT_CRYPTO_SALT", "test-only-crypto-salt")
