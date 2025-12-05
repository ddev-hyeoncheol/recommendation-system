from vespa.package import RankProfile


# ---------------------------------------------------------
# Default Rank Profile
# ---------------------------------------------------------
def get_default_rank_profile(vector_field_name: str, dimension: int) -> RankProfile:
    """
    Creates a default rank profile based on vector similarity.

    Args:
        vector_field_name (str): The name of the vector field.
        dimension (int): The dimension of the vector.

    Returns:
        RankProfile: The default rank profile.
    """
    return RankProfile(
        name="default",
        inputs=[("query(q)", f"tensor<float>(x[{dimension}])")],
        # First-phase ranking: Calculate closeness score
        first_phase=f"closeness(field, {vector_field_name})",
    )
