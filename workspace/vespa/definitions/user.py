from vespa.package import Schema, Field, ImportedField, Document, DocumentSummary, Summary, HNSW
from .common import get_default_rank_profile


# ---------------------------------------------------------
# User Data Schema (Parent)
# ---------------------------------------------------------
def create_user_schema() -> Schema:
    """
    [Parent] Schema for User Metadata
    - Stores raw user data without vectors.
    - Marked as 'global_document' to be accessible from any node.

    Returns:
        Schema: The User Data Schema.
    """
    # Document Fields
    document_fields = [
        Field(name="uid", type="string", indexing=["attribute", "summary"]),
        Field(name="country", type="string", indexing=["attribute", "summary"]),
        Field(name="state", type="string", indexing=["attribute", "summary"]),
        Field(name="zipcode", type="string", indexing=["attribute", "summary"]),
        Field(name="segment_id", type="string", indexing=["attribute", "summary"]),
    ]

    # Document
    document = Document(fields=document_fields)

    # User Schema
    schema = Schema(name="user", document=document, global_document=True)

    return schema


# ---------------------------------------------------------
# User Vector Schema (Child)
# ---------------------------------------------------------
def create_user_vector_schema(vector_dimension: int) -> Schema:
    """
    [Child] Schema for User Vector
    - References 'user' to access metadata.
    - Contains the HNSW vector index.

    Args:
        vector_dimension (int): The dimension of the vector.

    Returns:
        Schema: The User Vector Schema.
    """
    # ANN Index Fields
    hnsw_index = HNSW(distance_metric="angular", max_links_per_node=32, neighbors_to_explore_at_insert=200)

    # Document Fields
    document_fields = [
        Field(name="user_ref", type="reference<user>", indexing=["attribute"]),
        Field(name="model_version", type="string", indexing=["attribute", "summary"]),
        Field(name="embedding", type=f"tensor<float>(x[{vector_dimension}])", indexing=["attribute", "index", "summary"], ann=hnsw_index),
    ]

    # Imported Fields from User Schema
    imported_fields = [
        ImportedField(name="uid", reference_field="user_ref", field_to_import="uid"),
        ImportedField(name="country", reference_field="user_ref", field_to_import="country"),
        ImportedField(name="state", reference_field="user_ref", field_to_import="state"),
        ImportedField(name="zipcode", reference_field="user_ref", field_to_import="zipcode"),
    ]

    # Document Summary Fields from User Schema
    user_summary_fields = [
        Summary(name="uid", type=None, fields=[("source", "uid")]),
        Summary(name="country", type=None, fields=[("source", "country")]),
        Summary(name="state", type=None, fields=[("source", "state")]),
        Summary(name="zipcode", type=None, fields=[("source", "zipcode")]),
    ]

    # Document
    document = Document(fields=document_fields)

    # Document Summary
    document_summary = DocumentSummary(name="user_summary", summary_fields=user_summary_fields)

    # Rank Profile
    default_rank_profile = get_default_rank_profile(embedding_field_name="embedding", vector_dimension=vector_dimension)

    # User Vector Schema
    schema = Schema(
        name="user_vector",
        document=document,
        imported_fields=imported_fields,
        document_summaries=[document_summary],
        rank_profiles=[default_rank_profile],
    )

    return schema


# ---------------------------------------------------------
# User Segmentation Schema (Standalone)
# ---------------------------------------------------------
def create_user_segment_schema(vector_dimension: int) -> Schema:
    """
    [Standalone] Schema for User Segmentation.
    - Stores user segmentation vectors.
    - Contains the HNSW vector index.

    Args:
        vector_dimension (int): The dimension of the vector.

    Returns:
        Schema: The User Segmentation Schema.
    """
    # ANN Index Fields
    hnsw_index = HNSW(distance_metric="angular", max_links_per_node=32, neighbors_to_explore_at_insert=200)

    # Document Fields
    document_fields = [
        Field(name="segment_id", type="string", indexing=["attribute", "summary"]),
        Field(name="embedding", type=f"tensor<float>(x[{vector_dimension}])", indexing=["attribute", "index", "summary"], ann=hnsw_index),
    ]

    # Document
    document = Document(fields=document_fields)

    # Rank Profile
    rank_profile = get_default_rank_profile(embedding_field_name="embedding", vector_dimension=vector_dimension)

    # User Segmentation Schema
    schema = Schema(name="user_segment", document=document, rank_profiles=[rank_profile])

    return schema
