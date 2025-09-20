#!/usr/bin/env python3
"""
Debug script to test router initialization
"""

import sys
import traceback

print("🔍 Starting router debug...")

try:
    print("📥 Importing Tamil router...")
    from routing.tamil_router import tamil_router
    print("✅ Tamil router imported successfully")
    print(f"Tamil router status: {tamil_router.health_check()}")
except Exception as e:
    print(f"❌ Tamil router failed: {e}")
    traceback.print_exc()

try:
    print("\n📥 Importing English router...")
    from routing.english_router import english_router
    print("✅ English router imported successfully")
    print(f"English router status: {english_router.health_check()}")
except Exception as e:
    print(f"❌ English router failed: {e}")
    traceback.print_exc()

print("\n🏁 Debug complete")