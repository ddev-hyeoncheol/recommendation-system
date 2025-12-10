from vespa.package import RankProfile


# ---------------------------------------------------------
# Default Rank Profile
# ---------------------------------------------------------
def get_default_rank_profile(embedding_field_name: str, vector_dimension: int) -> RankProfile:
    """
    Creates a default rank profile based on vector similarity.

    Args:
        embedding_field_name (str): The name of the embedding field.
        vector_dimension (int): The dimension of the vector.

    Returns:
        RankProfile: The default rank profile.
    """
    closeness_feature = f"closeness(field, {embedding_field_name})"

    return RankProfile(
        name="default",
        inputs=[("query(q)", f"tensor<float>(x[{vector_dimension}])")],
        # First-phase ranking: Calculate closeness score
        first_phase=closeness_feature,
        match_features=[closeness_feature],
    )
