"""
test_llm_invoke.py

Quick test to verify LLM client initialization and API invocation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arlc.config import get_config
from ingestion.utils import get_llm_client_and_model

def test_llm_initialization():
    """Test if LLM client initializes correctly."""
    print("=" * 70)
    print("🔍 LLM INITIALIZATION TEST")
    print("=" * 70)
    
    cfg = get_config()
    
    print(f"\n✅ Config loaded successfully")
    print(f"   Mock LLM mode: {cfg.mock_llm}")
    print(f"   LLM Model: {cfg.llm_model}")
    print(f"   Gemini API Key configured: {'Yes' if cfg.gemini_api_key else 'No'}")
    
    if cfg.mock_llm:
        print("\n⚠️  Mock LLM mode is enabled. Skipping real API test.")
        return False
    
    try:
        llm_client, active_model = get_llm_client_and_model(cfg)
        print(f"\n✅ LLM client initialized successfully")
        print(f"   Client type: {type(llm_client).__name__}")
        print(f"   Active model: {active_model}")
        
        # Try a simple API call
        print(f"\n🚀 Testing LLM API call...")
        response = llm_client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'LLM is working!' in one sentence."},
            ],
            temperature=0.0,
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ LLM API call succeeded")
        print(f"   Response: {answer}")
        print(f"\n{'=' * 70}")
        print("✅ LLM INVOKE TEST PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ LLM initialization or API call failed:")
        print(f"   Error: {e}")
        print(f"\n{'=' * 70}")
        print("❌ LLM INVOKE TEST FAILED")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_llm_initialization()
    sys.exit(0 if success else 1)
