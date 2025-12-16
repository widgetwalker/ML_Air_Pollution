import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------
# 1. Load environment file
# ----------------------------
load_dotenv()

# Make sure src folder is added to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import AirQualityCLI
from src.display_utils import print_error
from src.config import settings


# ----------------------------
# 2. CLI main function
# ----------------------------
def main():
    try:
        # Validate API key for configured provider
        if not settings.validate_api_key():
            print(f"⚠ Warning: {settings.llm_provider.upper()} API key not configured properly")
            print(f"   The system will use fallback rule-based advice.")
        else:
            print(f"✓ Using {settings.llm_provider.upper()} as LLM provider")
        
        cli = AirQualityCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ----------------------------
# 3. Entry point
# ----------------------------
if __name__ == "__main__":
    main()