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
    closeness_feature = f"closeness(field, {embedding_field_name})"
    first_phase_ranking = FirstPhaseRanking(expression=closeness_feature)

    return RankProfile(
        name="default",
        inputs=[("query(q)", f"tensor<float>(x[{vector_dimension}])")],
        # First-phase ranking: Calculate closeness score
        first_phase=first_phase_ranking,
        match_features=[closeness_feature],
    )


# ---------------------------------------------------------
# Default Cold Start Rank Profile
# ---------------------------------------------------------
def get_default_cold_start_rank_profile() -> RankProfile:
    """
    Creates a default cold start rank profile based on rank.

    Returns:
        RankProfile: The default cold start rank profile.
    """
    first_phase_ranking = FirstPhaseRanking(expression="-attribute(rank)")

    return RankProfile(name="default", first_phase=first_phase_ranking)
