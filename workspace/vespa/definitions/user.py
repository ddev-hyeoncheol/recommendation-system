from vespa.package import Schema, Document, Field, ImportedField
from .common import get_default_rank_profile


# ---------------------------------------------------------
# User Data Schema (Parent)
# ---------------------------------------------------------
def create_user_data_schema() -> Schema:
    """
    [Parent] Schema for User Metadata
    - Stores raw user data without vectors.
    - Marked as 'global_document' to be accessible from any node.

    Returns:
        Schema: The user data schema.
    """
    return Schema(
        name="user_data",
        global_document=True,
        document=Document(
            fields=[
                Field(name="uid", type="string", indexing=["attribute", "summary"]),
                Field(name="country", type="string", indexing=["attribute", "summary"]),
                Field(name="state", type="string", indexing=["attribute", "summary"]),
                Field(name="zipcode", type="string", indexing=["attribute", "summary"]),
            ]
        ),
    )


# ---------------------------------------------------------
# User Vector Schema (Child)
# ---------------------------------------------------------
def create_user_schema(vector_dimension: int) -> Schema:
    """
    [Child] Schema for User Vector
    - References 'user_data' to access metadata.
    - Contains the HNSW vector index.

    Args:
        vector_dimension (int): The dimension of the vector.

    Returns:
        Schema: The user vector schema.
    """
    return Schema(
        name="user",
        document=Document(
            fields=[
                Field(
                    name="user_data_ref",
                    type="reference<user_data>",
                    indexing=["attribute"],
                ),
                Field(
                    name="user_vector",
                    type=f"tensor<float>(x[{vector_dimension}])",
                    indexing=["attribute", "index", "summary"],
                    attribute=["distance-metric: angular"],
                    index="hnsw",
                ),
            ]
        ),
        import_fields=[
            ImportedField(name="uid", reference_field="user_data_ref", field_to_import="uid"),
            ImportedField(name="country", reference_field="user_data_ref", field_to_import="country"),
            ImportedField(name="state", reference_field="user_data_ref", field_to_import="state"),
            ImportedField(name="zipcode", reference_field="user_data_ref", field_to_import="zipcode"),
        ],
        rank_profiles=[get_default_rank_profile("user_vector", vector_dimension)],
    )
