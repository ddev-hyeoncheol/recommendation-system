from fastapi import APIRouter, Depends
from typing import Dict, Any

from ..services import RecommendationService

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


# ---------------------------------------------------------
# Recommend Product (User -> Product)
# ---------------------------------------------------------
@router.get("/product/{uid}")
def recommend_product(
    uid: str,
    service: RecommendationService = Depends(RecommendationService),
) -> Dict[str, Any]:
    """
    Get a list of recommended products for a given user ID.

    Args:
        uid (str): The user ID to find recommendations for.
        service (RecommendationService): Dependency injected service.

    Returns:
        dict: Contains 'uid' and a list of 'recommendations' (pid, name, categories).
    """
    results = service.get_product_recommendations(uid)

    return {"uid": uid, "recommendations": results}


# ---------------------------------------------------------
# Recommend User (Product -> User)
# ---------------------------------------------------------
@router.get("/user/{pid}")
def recommend_user(
    pid: str,
    service: RecommendationService = Depends(RecommendationService),
) -> Dict[str, Any]:
    """
    Get a list of target users for a given product ID.

    Args:
        pid (str): The product ID to find potential buyers for.
        service (RecommendationService): Dependency injected service.

    Returns:
        dict: Contains 'pid' and a list of 'target_users' (uid, country, state, zipcode).
    """
    results = service.get_target_users(pid)

    return {"pid": pid, "target_users": results}
