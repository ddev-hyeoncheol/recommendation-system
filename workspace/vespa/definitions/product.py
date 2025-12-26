from vespa.package import Schema, Field, ImportedField, Document, DocumentSummary, Summary, HNSW
from .common import get_default_rank_profile


# ---------------------------------------------------------
# Product Data Schema (Parent)
# ---------------------------------------------------------
def create_product_schema() -> Schema:
    """
    [Parent] Schema for Product Metadata
    - Stores raw product data without vectors.
    - Marked as 'global_document' to be accessible from any node.

    Returns:
        Schema: The Product Data Schema.
    """
    # Document Fields
    document_fields = [
        Field(name="pid", type="string", indexing=["attribute", "summary"]),
        Field(name="name", type="string", indexing=["attribute", "summary"]),
        Field(name="categories", type="array<string>", indexing=["attribute", "summary"]),
    ]

    # Document
    document = Document(fields=document_fields)

    # Product Schema
    schema = Schema(name="product", document=document, global_document=True)

    return schema


# ---------------------------------------------------------
# Product Vector Schema (Child)
# ---------------------------------------------------------
def create_product_vector_schema(vector_dimension: int) -> Schema:
    """
    [Child] Schema for Product Vector
    - References 'product' to access metadata.
    - Contains the HNSW vector index.

    Args:
        vector_dimension (int): The dimension of the vector.

    Returns:
        Schema: The Product Vector Schema.
    """
    # ANN Index Fields
    hnsw_index = HNSW(distance_metric="angular", max_links_per_node=32, neighbors_to_explore_at_insert=200)

    # Document Fields
    document_fields = [
        Field(name="product_ref", type="reference<product>", indexing=["attribute"]),
        Field(name="model_version", type="string", indexing=["attribute", "summary"]),
        Field(name="embedding", type=f"tensor<float>(x[{vector_dimension}])", indexing=["attribute", "index", "summary"], ann=hnsw_index),
    ]

    # Imported Fields from Product Schema
    imported_fields = [
        ImportedField(name="pid", reference_field="product_ref", field_to_import="pid"),
        ImportedField(name="name", reference_field="product_ref", field_to_import="name"),
        ImportedField(name="categories", reference_field="product_ref", field_to_import="categories"),
    ]

    # Document Summary Fields from Product Schema
    product_summary_fields = [
        Summary(name="pid", type=None, fields=[("source", "pid")]),
        Summary(name="name", type=None, fields=[("source", "name")]),
        Summary(name="categories", type=None, fields=[("source", "categories")]),
    ]

    # Document
    document = Document(fields=document_fields)

    # Document Summary
    document_summary = DocumentSummary(name="product_summary", summary_fields=product_summary_fields)

    # Rank Profile
    rank_profile = get_default_rank_profile(embedding_field_name="embedding", vector_dimension=vector_dimension)

    # Product Vector Schema
    schema = Schema(
        name="product_vector",
        document=document,
        imported_fields=imported_fields,
        document_summaries=[document_summary],
        rank_profiles=[rank_profile],
    )

    return schema
