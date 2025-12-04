from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@router.get("")
def health_check():
    """
    Basic health check endpoint to verify server status.

    Returns:
        dict: Simple status message indicating the server is running.
    """
    return {"status": "Recommendation Service API Server is Running"}
