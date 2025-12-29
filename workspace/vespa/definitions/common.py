from vespa.package import RankProfile, FirstPhaseRanking


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
    ranking_expression = f"closeness(field, {embedding_field_name})"
    first_phase_ranking = FirstPhaseRanking(expression=ranking_expression)

    return RankProfile(
        name="default",
        inputs=[("query(q)", f"tensor<float>(x[{vector_dimension}])")],
        # First-phase ranking: Calculate closeness score
        first_phase=first_phase_ranking,
        match_features=[ranking_expression],
    )
