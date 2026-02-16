import requests
import json

API_KEY = "AIzaSyBIl0FszcAAYd7YAR8MPsmhDGLu8S2mVU0"

# Test 1: simple prompt with gemini-2.0-flash
print("=== Test 1: gemini-2.0-flash ===")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
resp = requests.post(url, json={
    "contents": [{"parts": [{"text": "Say hello in JSON format: {\"message\": \"hello\"}"}]}],
    "generationConfig": {
        "temperature": 0.2,
        "maxOutputTokens": 256,
        "responseMimeType": "application/json"
    }
}, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Body: {resp.text[:500]}")

# Test 2: list available models
print("\n=== Test 2: List models ===")
resp2 = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}", timeout=30)
if resp2.status_code == 200:
    models = resp2.json().get("models", [])
    flash_models = [m["name"] for m in models if "flash" in m["name"].lower() or "gemini-2" in m["name"].lower()]
    print(f"Found {len(models)} models total")
    print(f"Flash/Gemini-2+ models: {json.dumps(flash_models, indent=2)}")
else:
    print(f"Status: {resp2.status_code}")
    print(f"Body: {resp2.text[:500]}")
