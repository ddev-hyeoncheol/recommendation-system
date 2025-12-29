import numpy as np

from fastapi import Depends, HTTPException
from typing import List, Dict, Any, Optional
from vespa.application import Vespa
from redis import Redis

from ..config import Settings, get_settings
from ..vespa_client import get_vespa_client
from ..redis_client import get_redis_client


# ---------------------------------------------------------
# Recommendation Service
# ---------------------------------------------------------
class RecommendationService:
    """
    Service layer for handling recommendation logic using Vespa.
    Manages vector retrieval, nearest neighbor search, and public APIs for product and user recommendations.

    Attributes:
        settings (Settings): The application settings
        vespa_client (Vespa): The Vespa client for querying Vespa
        redis_client (Redis): The Redis client for Redis operations
    """

    def __init__(
        self,
        settings: Settings = Depends(get_settings),
        vespa_client: Vespa = Depends(get_vespa_client),
        redis_client: Redis = Depends(get_redis_client),
    ):
        self.settings = settings
        self.vespa_client = vespa_client
        self.redis_client = redis_client

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

            response = self.vespa_client.query(body=body)
            return response.hits

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vespa Query Error: {str(e)}")

    # ---------------------------------------------------------
    # Fetch Vector
    # ---------------------------------------------------------
    def _fetch_vector(self, doc_type: str, id_value: str, model_version: str = None) -> List[float]:
        """
        Fetch the vector for a given document ID from the Child Vector Schema.

        Args:
            doc_type (str): The type of the document ("user" or "product").
            id_value (str): The value of the document ID (uid or pid).
            model_version (str): The model version. Defaults to the latest model version.

        Returns:
            List[float]: The embedding vector for the given document ID. None otherwise.
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
    # Fetch Segment Vector
    # ---------------------------------------------------------
    def _fetch_segment_vector(self, doc_type: str, segment_id: str) -> List[float]:
        """
        Fetch the segment vector for a given segment document ID from the Segment Schema.

        Args:
            doc_type (str): The type of the document ("user" or "product").
            segment_id (str): The value of the segment document ID.

        Returns:
            List[float]: The embedding vector for the given segment document ID. None otherwise.
        """
        segment_schema = f"{doc_type}_segment"

        yql = f"select embedding from {segment_schema} where segment_id contains '{segment_id}'"

        hits = self._query_vespa(yql=yql, hits=1)

        if hits:
            return hits[0]["fields"]["embedding"]["values"]

        return None

    # ---------------------------------------------------------
    # Fetch Metadata
    # ---------------------------------------------------------
    def _fetch_metadata(self, doc_type: str, id_value: str) -> Dict[str, Any]:
        """
        Fetch the metadata for a given document ID from the Parent Metadata Schema.

        Args:
            doc_type (str): The type of the document ("user" or "product").
            id_value (str): The value of the document ID (uid or pid).

        Returns:
            Dict[str, Any]: The metadata for the given document ID.
        """
        metadata_schema = f"{doc_type}"
        id_field = "uid" if doc_type == "user" else "pid"

        yql = f"select * from {metadata_schema} where {id_field} contains '{id_value}'"

        hits = self._query_vespa(yql=yql, hits=1)

        if hits:
            return hits[0]["fields"]

        raise HTTPException(status_code=404, detail=f"Document '{id_value}' not found in {metadata_schema} schema")

    # ---------------------------------------------------------
    # Cache User Session Recommendations
    # ---------------------------------------------------------
    def _get_recent_interactions(self, doc_type: str, id_value: str) -> List[str]:
        """
        Get the recent interactions for a given user ID from Redis session.

        Args:
            doc_type (str): The type of the document ("user" or "product").
            id_value (str): The value of the document ID (uid or pid).

        Returns:
            List[str]: The recent interactions for the given document ID.
        """
        redis_key = f"{doc_type}:session:recent_interactions:{id_value}"

        try:
            recent_interactions = self.redis_client.lrange(redis_key, 0, 9)
            return recent_interactions if recent_interactions else []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")

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
    # Compute Real-Time Vector
    # ---------------------------------------------------------
    def _compute_realtime_vector(self, base_vector: List[float], interaction_type: str, recent_interactions: List[str], model_version: str = None) -> List[float]:
        """
        Compute the real-time vector for a given base vector and recent interactions.

        Args:
            base_vector (List[float]): The base vector.
            interaction_type (str): The type of interaction ("user" or "product").
            recent_interactions (List[str]): The recent interactions.
            model_version (str): The model version. Defaults to the latest model version.

        Returns:
            List[float]: The real-time vector.
        """
        if not recent_interactions:
            return base_vector

        target_schema = f"{interaction_type}_vector"
        id_field = "uid" if interaction_type == "user" else "pid"
        model_version = model_version if model_version else self.settings.latest_model_version
        ids_string = ", ".join([f"'{interaction_id}'" for interaction_id in recent_interactions])

        yql = f"select embedding from {target_schema} where {id_field} in ({ids_string}) and model_version contains '{model_version}'"

        raw_hits = self._query_vespa(yql=yql, hits=len(recent_interactions))

        vectors = [hit["fields"]["embedding"]["values"] for hit in raw_hits if hit["fields"]["embedding"]]

        if not vectors:
            return base_vector

        def normalize_vector(vector: List[float]) -> List[float]:
            norm = np.linalg.norm(vector)
            return vector / norm if norm > 0 else vector

        recent_vector = np.mean(vectors, axis=0)
        recent_vector = normalize_vector(recent_vector)

        alpha = self.settings.recommend_alpha
        beta = self.settings.recommend_beta

        combined_vector = (alpha * np.array(base_vector)) + (beta * recent_vector)
        combined_vector = normalize_vector(combined_vector)

        return combined_vector.tolist()

    # ---------------------------------------------------------
    # Generate Product Recommendations (Public API)
    # ---------------------------------------------------------
    def get_product_recommendations(self, uid: str) -> List[Dict[str, Any]]:
        """
        Generates product recommendations for a specific user.

        Flow:
        1. Fetch the embedding vector for the user.
        2. Get the recent interactions for the user.
        3-1. If recent interactions exist, compute the real-time vector using the recent interactions.
        3-2. If recent interactions do not exist, fetch the segment vector using the user metadata.
        4. Perform a nearest neighbor search for the product embedding vector using the real-time vector.
        5. Return the results with product metadata fields (pid, name, categories).

        Args:
            uid (str): The user ID.

        Returns:
            List[Dict[str, Any]]: List of recommended products (pid, name, categories).
        """
        base_vector = self._fetch_vector(doc_type="user", id_value=uid)
        recent_interactions = self._get_recent_interactions(doc_type="user", id_value=uid)
        print(recent_interactions)

        if not base_vector:
            # Cold Start : Fetch the segment vector for the user.
            user_metadata = self._fetch_metadata(doc_type="user", id_value=uid)
            segment_id = user_metadata.get("segment_id")

            if not segment_id:
                return []

            base_vector = self._fetch_segment_vector(doc_type="user", segment_id=segment_id)
            if not base_vector:
                raise HTTPException(status_code=404, detail=f"Segment Document '{segment_id}' not found in user_segment schema.")

        user_vector = self._compute_realtime_vector(base_vector=base_vector, interaction_type="product", recent_interactions=recent_interactions)

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
        3. Return the results with user metadata fields (uid, country, state, zipcode).

        Args:
            pid (str): The product ID.

        Returns:
            List[Dict[str, Any]]: List of target users (uid, country, state, zipcode).
        """
        product_vector = self._fetch_vector(doc_type="product", id_value=pid)

        results = self._search_nearest(target_doc_type="user", query_vector=product_vector)

        return [{"uid": r.get("uid"), "country": r.get("country"), "state": r.get("state"), "zipcode": r.get("zipcode")} for r in results]
