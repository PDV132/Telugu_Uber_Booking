# Safe mock implementations for testing

def speech_to_text(audio_path: str) -> str:
    print(f"[MOCK] Converting audio '{audio_path}' to text...")
    return "Simulated speech-to-text result"

def language_detection(text: str) -> str:
    print(f"[MOCK] Detecting language for text: '{text}'")
    return "en"  # simulate English

def translation(text: str) -> str:
    print(f"[MOCK] Translating text to English: '{text}'")
    return "Simulated translation to English"

def intent_recognition(text: str) -> str:
    print(f"[MOCK] Recognizing intent for text: '{text}'")
    return "book"  # simulate ride booking intent

def entity_extraction(text: str) -> dict:
    print(f"[MOCK] Extracting entities from text: '{text}'")
    return {
        "pickup": "123 Main St",
        "drop": "456 Park Ave",
        "payment_mode": "card"
    }

def check_consent(user_id: str) -> bool:
    print(f"[MOCK] Checking consent for user {user_id}")
    return True

def notify_user(user_id: str, message: str):
    print(f"[MOCK] Notifying user {user_id}: {message}")

def book_ride(pickup: str, drop: str, payment_mode: str) -> dict:
    print(f"[MOCK] Simulating ride booking from {pickup} to {drop} via {payment_mode}")
    return {
        "ride_id": "RIDE123",
        "driver": "Test Driver",
        "eta": 5
    }
