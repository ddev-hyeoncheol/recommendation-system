from vespa.package import Schema, Document, Field, ImportedField
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
        Schema: The user data schema.
    """
    schema = Schema(name="user", document=Document(), global_document=True)

    # Add fields for user metadata
    schema.add_fields(
        Field(name="uid", type="string", indexing=["attribute", "summary"]),
        Field(name="country", type="string", indexing=["attribute", "summary"]),
        Field(name="state", type="string", indexing=["attribute", "summary"]),
        Field(name="zipcode", type="string", indexing=["attribute", "summary"]),
    )

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
        Schema: The user vector schema.
    """

    schema = Schema(name="user_vector", document=Document())

    # Add fields for user vector
    schema.add_fields(
        Field(name="user_ref", type="reference<user>", indexing=["attribute"]),
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
    schema.add_imported_field(ImportedField(name="uid", reference_field="user_ref", field_to_import="uid"))
    schema.add_imported_field(ImportedField(name="country", reference_field="user_ref", field_to_import="country"))
    schema.add_imported_field(ImportedField(name="state", reference_field="user_ref", field_to_import="state"))
    schema.add_imported_field(ImportedField(name="zipcode", reference_field="user_ref", field_to_import="zipcode"))

    # Add default rank profile
    schema.add_rank_profile(get_default_rank_profile(embedding_field_name="embedding", vector_dimension=vector_dimension))

    return schema
