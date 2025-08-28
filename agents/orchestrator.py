# agents/orchestrator.py
import asyncio

# -------------------------------
# Mock Tools (Simulated for safety)
# -------------------------------
def speech_to_text(audio_path):
    return f"Transcribed text from {audio_path}"

def detect_language(text):
    return "te" if "telugu" in text.lower() else "en"

def translate_to_english(text):
    return text + " (translated to English)"

def recognize_intent(text):
    if "book" in text.lower():
        return "book"
    return "unknown"

def extract_entities(text):
    return {"pickup": "LocationA", "drop": "LocationB", "payment_mode": "cash"}

def check_consent(user_id):
    return True

def notify_user(user_id, message):
    print(f"[Notification] To {user_id}: {message}")

def book_ride(pickup, drop, payment_mode):
    print(f"[Simulation] Ride booked from {pickup} to {drop} with {payment_mode}")
    return {"ride_id": "SIM123", "driver": "SimDriver", "eta": 5}


# -------------------------------
# Mock Agent Class
# -------------------------------
class MockAgent:
    async def ainvoke(self, inputs: dict):
        user_input = inputs.get("input", "")
        if "audio" in user_input:
            text = speech_to_text(user_input)
            lang = detect_language(text)
            if lang == "te" and check_consent("user123"):
                text = translate_to_english(text)
            intent = recognize_intent(text)
            entities = extract_entities(text)
            if intent == "book":
                ride = book_ride(**entities)
                notify_user("user123", f"Ride ready! Driver: {ride['driver']}, ETA: {ride['eta']} mins")
                return f"Simulated booking complete for {entities['pickup']} → {entities['drop']}"
            else:
                return f"Processed audio input: {text}"
        else:
            return f"No audio provided. Received text input: {user_input}"


# -------------------------------
# Exported agent object
# -------------------------------
agent_graph = MockAgent()
