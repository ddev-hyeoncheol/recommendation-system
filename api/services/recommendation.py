from fastapi import Depends, HTTPException
from typing import List, Dict, Any, Optional
from vespa.application import Vespa

from ..config import Settings, get_settings
from ..vespa_client import get_vespa_client


# ---------------------------------------------------------
# Recommendation Service
# ---------------------------------------------------------
class RecommendationService:
    """
    Service layer for handling recommendation logic using Vespa.
    Manages vector retrieval, nearest neighbor search, and public APIs for product and user recommendations.

    Attributes:
        settings (Settings): The application settings
        client (Vespa): The Vespa client for querying Vespa
    """

    def __init__(
        self,
        settings: Settings = Depends(get_settings),
        client: Vespa = Depends(get_vespa_client),
    ):
        self.settings = settings
        self.client = client

    # ---------------------------------------------------------
    # Base Query Executor
    # ---------------------------------------------------------
    def _query_vespa(self, yql: str, hits: int = 1, body_params: dict = None) -> list:
        """
        Execute a Vespa YQL query against Vespa and return the hits from the response.

        Args:
            yql (str): The YQL query to execute
            hits (int): The number of hits to return
            body_params (dict): Additional body parameters to include in the query

        Returns:
            list: A list of hits (Documents) from the Vespa response

        Raises:
            HTTPException: If the Vespa query execution fails (500 Internal Server Error)
        """
        try:
            body = {"yql": yql, "hits": hits, "presentation.format": "json"}

            if body_params:
                body.update(body_params)

            response = self.client.query(body=body)
            return response.hits

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vespa Query Error: {str(e)}")

    # ---------------------------------------------------------
    # Fetch Vector
    # ---------------------------------------------------------
    def _fetch_vector(self, doc_type: str, id_value: str, model_version: str = None) -> list:
        """
        Fetch the vector for a given document ID from the Child Vector Schema

        Args:
            doc_type (str): The type of the document ("user" or "product")
            id_value (str): The value of the document ID (uid or pid)
            model_version (str): The model version. Defaults to the latest model version.

        Returns:
            list: The embedding vector for the given document ID. None otherwise.
        """
        vector_schema = f"{doc_type}_vector"
        id_field = "uid" if doc_type == "user" else "pid"
        model_version = model_version if model_version else self.settings.latest_model_version

        yql = f"select embedding from {vector_schema} where {id_field} contains '{id_value}' and model_version contains '{model_version}'"

        hits = self._query_vespa(yql=yql, hits=1)

        if hits:
            return hits[0]["fields"]["embedding"]["values"]

        return None

    # ---------------------------------------------------------
    # Search Nearest Neighbors
    # ---------------------------------------------------------
    def _search_nearest(
        self,
        target_doc_type: str,
        query_vector: List[float],
        hits: Optional[int] = None,
        target_hits: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs an Approximate Nearest Neighbor (ANN) search using the HNSW index.

        Args:
            target_doc_type (str): The type of the target document ("user" or "product")
            query_vector (List[float]): The embedding vector of the query document.
            hits (Optional[int]): The number of hits to return. Defaults to the setting value.
            target_hits (Optional[int]): The number of hits to return for the target document. Defaults to the setting value.

        Returns:
            List[Dict[str, Any]]: A list of nearest neighbor search results with metadata fields.
        """
        target_schema = f"{target_doc_type}_vector"
        summary_name = f"{target_doc_type}_summary"

        hits = hits if hits is not None else self.settings.recommend_hits
        target_hits = target_hits if target_hits is not None else self.settings.recommend_target_hits

        yql = f"select * from {target_schema} where {{targetHits:{target_hits}}}nearestNeighbor(embedding, q)"

        body_params = {
            "ranking": "default",
            "ranking.features.query(q)": query_vector,
            "summary": summary_name,
        }

        raw_hits = self._query_vespa(yql=yql, hits=hits, body_params=body_params)
        return [hit.get("fields", {}) for hit in raw_hits]

    # ---------------------------------------------------------
    # Generate Product Recommendations (Public API)
    # ---------------------------------------------------------
    def get_product_recommendations(self, uid: str) -> List[Dict[str, Any]]:
        """
        Generates product recommendations for a specific user.

        Flow:
        1. Fetch the embedding vector for the user.
        2. Perform a nearest neighbor search for the product embedding vector using the user vector.
        3. Return the results with metadata fields (pid, name, categories).

        Args:
            uid (str): The user ID.

        Returns:
            List[Dict[str, Any]]: List of recommended products (pid, name, categories).
        """
        user_vector = self._fetch_vector(doc_type="user", id_value=uid)
        results = self._search_nearest(target_doc_type="product", query_vector=user_vector)

        return [{"pid": r.get("pid"), "name": r.get("name"), "categories": r.get("categories")} for r in results]

    # ---------------------------------------------------------
    # Generate Target Users (Public API)
    # ---------------------------------------------------------
    def get_target_users(self, pid: str) -> List[Dict[str, Any]]:
        """
        Generates target users for a specific product.

        Flow:
        1. Fetch the embedding vector for the product.
        2. Perform a nearest neighbor search for the user embedding vector using the product vector.
        3. Return the results with metadata fields (uid, country, state, zipcode).

        Args:
            pid (str): The product ID.

        Returns:
            List[Dict[str, Any]]: List of target users (uid, country, state, zipcode).
        """
        product_vector = self._fetch_vector(doc_type="product", id_value=pid)
        results = self._search_nearest(target_doc_type="user", query_vector=product_vector)

        return [{"uid": r.get("uid"), "country": r.get("country"), "state": r.get("state"), "zipcode": r.get("zipcode")} for r in results]
