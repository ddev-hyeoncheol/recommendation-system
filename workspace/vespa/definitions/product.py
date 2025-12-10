from vespa.package import Schema, Document, Field, ImportedField
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
        Schema: The product data schema.
    """
    schema = Schema(name="product", document=Document(), global_document=True)

    # Add fields for product metadata
    schema.add_fields(
        Field(name="pid", type="string", indexing=["attribute", "summary"]),
        Field(name="name", type="string", indexing=["attribute", "summary"]),
        Field(name="categories", type="array<string>", indexing=["attribute", "summary"]),
    )

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
        Schema: The product vector schema.
    """
    schema = Schema(name="product_vector", document=Document())

    # Add fields for product vector
    schema.add_fields(
        Field(name="product_ref", type="reference<product>", indexing=["attribute"]),
        Field(name="model_version", type="string", indexing=["attribute", "summary"]),
        Field(
            name="embedding",
            type=f"tensor<float>(x[{vector_dimension}])",
            indexing=["attribute", "index", "summary"],
            attribute=["distance-metric: angular"],
            index="hnsw",
        ),
    )

    # Add imported fields
    schema.add_imported_field(ImportedField(name="pid", reference_field="product_ref", field_to_import="pid"))
    schema.add_imported_field(ImportedField(name="name", reference_field="product_ref", field_to_import="name"))
    schema.add_imported_field(ImportedField(name="categories", reference_field="product_ref", field_to_import="categories"))

    # Add default rank profile
    schema.add_rank_profile(get_default_rank_profile(embedding_field_name="embedding", vector_dimension=vector_dimension))

    return schema
