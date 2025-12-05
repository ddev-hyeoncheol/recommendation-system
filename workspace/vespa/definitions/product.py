from vespa.package import Schema, Document, Field, ImportedField
from .common import get_default_rank_profile


# ---------------------------------------------------------
# Product Data Schema
# ---------------------------------------------------------
def create_product_data_schema() -> Schema:
    """
    [Parent] Schema for Product Metadata
    - Stores raw product data without vectors.
    - Marked as 'global_document' to be accessible from any node.

    Returns:
        Schema: The product data schema.
    """
    return Schema(
        name="product_data",
        global_document=True,
        document=Document(
            fields=[
                Field(name="pid", type="string", indexing=["attribute", "summary"]),
                Field(
                    name="name",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="categories",
                    type="array<string>",
                    indexing=["attribute", "summary"],
                ),
            ]
        ),
    )


# ---------------------------------------------------------
# Product Vector Schema
# ---------------------------------------------------------
def create_product_schema(vector_dimension: int) -> Schema:
    """
    [Child] Schema for Product Vector
    - References 'product_data' to access metadata.
    - Contains the HNSW vector index.

    Args:
        vector_dimension (int): The dimension of the vector.

    Returns:
        Schema: The product vector schema.
    """
    return Schema(
        name="product",
        document=Document(
            fields=[
                Field(
                    name="product_data_ref",
                    type="reference<product_data>",
                    indexing=["attribute"],
                ),
                Field(
                    name="product_vector",
                    type=f"tensor<float>(x[{vector_dimension}])",
                    indexing=["attribute", "index", "summary"],
                    attribute=["distance-metric: angular"],
                    index="hnsw",
                ),
            ]
        ),
        import_fields=[
            ImportedField(name="pid", reference_field="product_data_ref", field_to_import="pid"),
            ImportedField(name="name", reference_field="product_data_ref", field_to_import="name"),
            ImportedField(name="categories", reference_field="product_data_ref", field_to_import="categories"),
        ],
        rank_profiles=[get_default_rank_profile("product_vector", vector_dimension)],
    )
