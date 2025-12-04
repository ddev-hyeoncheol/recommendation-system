from functools import lru_cache
from vespa.application import Vespa

from .config import settings


# ---------------------------------------------------------
# Vespa Client Provider
# ---------------------------------------------------------
@lru_cache()
def get_vespa_client() -> Vespa:
    """
    Creates and returns a cached Vespa client instance.
    Uses lru_cache to ensure a singleton pattern for the client.

    Returns:
        Vespa: Configured Vespa client instance ready for queries.
    """
    # Construct the base URL with protocol
    vespa_url = f"http://{settings.vespa_host}"
    vespa_port = settings.vespa_port

    # Create client instance
    # Note: connection is established lazily when queries are executed
    vespa_client = Vespa(url=vespa_url, port=vespa_port)

    return vespa_client
