import os
import shutil

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

from vespa.package import ApplicationPackage
from definitions.user import create_user_schema, create_user_vector_schema
from definitions.product import create_product_schema, create_product_vector_schema

# ---------------------------------------------------------
# Configuration & Environment Setup
# ---------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Vespa Configuration
VESPA_APP_NAME = os.getenv("VESPA_APP_NAME")
APP_PACKAGE_DIR = Path(os.getenv("APP_PACKAGE_DIR"))

# Matrix Factorization Hyperparameter
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION"))


# ---------------------------------------------------------
# Validation Override Helper
# ---------------------------------------------------------
def create_validation_overrides() -> str:
    """
    Generates XML content to allow destructive schema changes.

    Returns:
        str: The XML content to allow destructive schema changes.
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
    <allow until="{until_date}">indexing-change</allow>
</validation-overrides>
"""


# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
def main():
    print(f"🔨 Creating Vespa Application Package")
    print(f"📂 Target Directory: {APP_PACKAGE_DIR}")

    # Clean output directory
    if APP_PACKAGE_DIR.exists():
        shutil.rmtree(APP_PACKAGE_DIR)

    # Define Schemas
    schemas = [
        create_user_schema(),
        create_product_schema(),
        create_user_vector_schema(VECTOR_DIMENSION),
        create_product_vector_schema(VECTOR_DIMENSION),
    ]

    # Create Application Package
    app_package = ApplicationPackage(name=VESPA_APP_NAME, schema=schemas)

    # Export to files
    app_package.to_files(str(APP_PACKAGE_DIR))

    # Add validation-overrides.xml manually
    with open(APP_PACKAGE_DIR / "validation-overrides.xml", "w") as f:
        validation_overrides = create_validation_overrides().strip()
        f.write(validation_overrides)

    print(f"✅ Package generated successfully at: {APP_PACKAGE_DIR}")


if __name__ == "__main__":
    main()
