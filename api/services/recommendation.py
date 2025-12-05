from fastapi import Depends, HTTPException
from vespa.application import Vespa

from ..config import Settings, get_settings
from ..vespa_client import get_vespa_client


# ---------------------------------------------------------
# Recommendation Service
# ---------------------------------------------------------
class RecommendationService:
    def __init__(
        self,
        settings: Settings = Depends(get_settings),
        client: Vespa = Depends(get_vespa_client),
    ):
        self.settings = settings
        self.client = client

    # ---------------------------------------------------------
    # Fetch Vector
    # ---------------------------------------------------------
    def _fetch_vector(self, schema: str, id_field: str, doc_id: str, vector_field: str) -> list:
        """
        Fetch the vector for a given document ID

        Args:
            schema (str): The schema name
            id_field (str): The field name for the document ID
            doc_id (str): The document ID
            vector_field (str): The field name for the vector

        Returns:
            list: The vector for the given document ID
        """
        try:
            yql = f"select * from {schema} where {id_field} = '{doc_id}'"
            body = {"yql": yql, "hits": 1, "presentation.format": "json"}

            res = self.client.query(body=body)

            if not res.hits:
                raise HTTPException(
                    status_code=404,
                    detail=f"{schema.capitalize()} '{doc_id}' not found",
                )

            hit = res.hits[0]
            if "fields" not in hit:
                raise HTTPException(status_code=500, detail="Vespa response missing 'fields'")

            if vector_field not in hit["fields"]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Vector field '{vector_field}' missing in document",
                )

            return hit["fields"][vector_field]["values"]

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vespa Error: {str(e)}")

    # ---------------------------------------------------------
    # Search Nearest
    # ---------------------------------------------------------
    def _search_nearest(
        self,
        target_schema: str,
        vector_field: str,
        query_vector: list,
        hits: int = None,
    ) -> list:
        """
        Perform a nearest neighbor search using the vector

        Args:
            target_schema (str): The schema name
            vector_field (str): The field name for the vector
            query_vector (list): The query vector
            hits (int): The number of hits to return. Defaults to setting value.

        Returns:
            list: The nearest neighbor search results
        """
        # Use configured defaults if not provided
        final_hits = hits if hits is not None else self.settings.recommend_hits
        target_hits = self.settings.recommend_target_hits

        try:
            yql = f"select * from {target_schema} where {{targetHits:{target_hits}}}nearestNeighbor({vector_field}, q)"
            body = {
                "yql": yql,
                "hits": final_hits,
                "ranking": "default",
                "ranking.features.query(q)": query_vector,
                "presentation.format": "json",
            }

            res = self.client.query(body=body)

            return [hit.get("fields", {}) for hit in res.hits]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

    # ---------------------------------------------------------
    # Get Product Recommendations
    # ---------------------------------------------------------
    def get_product_recommendations(self, uid: str) -> list:
        """
        Get a list of recommended products for a given user ID.

        Args:
            uid (str): The user ID.

        Returns:
            list: List of recommended products (pid, name, categories).
        """
        # 1. Fetch User Vector
        user_vector = self._fetch_vector(schema="user", id_field="uid", doc_id=uid, vector_field="user_vector")

        # 2. Search Nearest Products
        raw_results = self._search_nearest(
            target_schema="product",
            vector_field="product_vector",
            query_vector=user_vector,
        )

        # 3. Format Response
        return [
            {
                "pid": r.get("pid"),
                "name": r.get("name"),
                "categories": r.get("categories", []),  # Provide default list
            }
            for r in raw_results
        ]

    # ---------------------------------------------------------
    # Get Target Users
    # ---------------------------------------------------------
    def get_target_users(self, pid: str) -> list:
        """
        Get a list of target users for a given product ID.

        Args:
            pid (str): The product ID.

        Returns:
            list: List of target users (uid, country, state, zipcode).
        """
        # 1. Fetch Product Vector
        product_vector = self._fetch_vector(schema="product", id_field="pid", doc_id=pid, vector_field="product_vector")

        # 2. Search Nearest Users
        raw_results = self._search_nearest(
            target_schema="user",
            vector_field="user_vector",
            query_vector=product_vector,
        )

        # 3. Format Response
        return [
            {
                "uid": r.get("uid"),
                "country": r.get("country"),
                "state": r.get("state"),
                "zipcode": r.get("zipcode"),
            }
            for r in raw_results
        ]
