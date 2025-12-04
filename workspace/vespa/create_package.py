import os
import shutil

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    RankProfile
)

# ---------------------------------------------------------
# 1. Configuration & Environment Setup
# ---------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Vespa Configuration
VESPA_APP_NAME = os.getenv("VESPA_APP_NAME")
APP_PACKAGE_DIR = Path(os.getenv("APP_PACKAGE_DIR"))

# Model Configuration
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION"))


# ---------------------------------------------------------
# 2. Schema Helpers (Data Structure Definition)
# ---------------------------------------------------------
def get_default_rank_profile(vector_field_name: str, dimension: int) -> RankProfile:
    """
    Creates a default rank profile based on vector similarity.

    Args:
        vector_field_name (str): The name of the vector field (e.g., 'user_vector').
        dimension (int): The dimension of the vector.

    Returns:
        RankProfile: Configured rank profile with closeness matching.
    """
    return RankProfile(
        name="default",
        inputs=[("query(q)", f"tensor<float>(x[{dimension}])")],
        # First-phase ranking: Calculate closeness score
        first_phase=f"closeness(field, {vector_field_name})"
    )


def create_user_schema() -> Schema:
    """
    Defines the Schema for 'User' documents.
    Includes metadata fields and a vector field for HNSW indexing.
    """
    return Schema(
        name="user",
        document=Document(
            fields=[
                Field(name="uid", type="string",
                      indexing=["attribute", "summary"]),
                Field(name="country", type="string",
                      indexing=["attribute", "summary"]),
                Field(name="state", type="string",
                      indexing=["attribute", "summary"]),
                Field(name="zipcode", type="string",
                      indexing=["attribute", "summary"]),

                # User Vector Field (HNSW Index)
                Field(
                    name="user_vector",
                    type=f"tensor<float>(x[{VECTOR_DIMENSION}])",
                    indexing=["attribute", "index", "summary"],
                    attribute=["distance-metric: angular"],
                    index="hnsw"
                ),
            ]
        ),
        rank_profiles=[get_default_rank_profile(
            "user_vector", VECTOR_DIMENSION)],
    )


def create_product_schema() -> Schema:
    """
    Defines the Schema for 'Product' documents.
    Includes metadata fields and a vector field for HNSW indexing.
    """
    return Schema(
        name="product",
        document=Document(
            fields=[
                Field(name="pid", type="string",
                      indexing=["attribute", "summary"]),
                Field(
                    name="name",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="categories",
                    type="array<string>",
                    indexing=["attribute", "summary"],
                ),

                # Product Vector Field (HNSW Index)
                Field(
                    name="product_vector",
                    type=f"tensor<float>(x[{VECTOR_DIMENSION}])",
                    indexing=["attribute", "index", "summary"],
                    attribute=["distance-metric: angular"],
                    index="hnsw",
                ),
            ]
        ),
        rank_profiles=[get_default_rank_profile(
            "product_vector", VECTOR_DIMENSION)],
    )


# ---------------------------------------------------------
# 3. Validation Override Helper
# ---------------------------------------------------------
def create_validation_overrides() -> str:
    """
    Generates XML content to allow destructive schema changes (e.g., field removal).
    Valid for 7 days from execution.
    """
    until_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    return f"""
<validation-overrides>
    <allow until="{until_date}">indexing-mode-change</allow>
    <allow until="{until_date}">field-type-change</allow>
    <allow until="{until_date}">tensor-type-change</allow>
    <allow until="{until_date}">resource-limits</allow>
    <allow until="{until_date}">content-removal</allow>
    <allow until="{until_date}">index-mode-change</allow>
</validation-overrides>
"""


# ---------------------------------------------------------
# 4. Main Execution Flow
# ---------------------------------------------------------
def main():
    print(f"🔨 Creating Vespa Application Package...")
    print(f"📂 Target Directory: {APP_PACKAGE_DIR}")

    # 1. Clean output directory
    if APP_PACKAGE_DIR.exists():
        shutil.rmtree(APP_PACKAGE_DIR)

    # 2. Define Schemas
    user_schema = create_user_schema()
    product_schema = create_product_schema()

    # 3. Create Application Package
    app_package = ApplicationPackage(
        name=VESPA_APP_NAME,
        schema=[user_schema, product_schema]
    )

    # 4. Export to files (services.xml, schemas/*.sd)
    app_package.to_files(str(APP_PACKAGE_DIR))

    # 5. Add validation-overrides.xml manually
    with open(APP_PACKAGE_DIR / "validation-overrides.xml", "w") as f:
        validation_overrides = create_validation_overrides().strip()
        f.write(validation_overrides)

    print(f"✅ Package generated successfully at: {APP_PACKAGE_DIR}")


if __name__ == "__main__":
    main()
