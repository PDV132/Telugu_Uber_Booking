import asyncio
from agents.orchestrator import agent_graph

async def run_simulation(audio_path: str, user_id: str):
    # Case 1: No audio file
    if not audio_path:
        print("No audio provided. Please type your request instead.")
        user_input = f"Simulated typed input from {user_id}"
        result = await agent_graph.run({"input": user_input})
        print(f"[Simulation result] {result}")
        return

    # Case 2: Audio file provided
    print(f"Processing audio file: {audio_path}")

    # Simulate the steps sequentially
    audio_text = agent_graph.tools["SpeechToText"].func(audio_path)
    lang = agent_graph.tools["DetectLanguage"].func(audio_text)

    if lang == "te":  # Simulate translation if Telugu
        text = agent_graph.tools["TranslateToEnglish"].func(audio_text)
    else:
        text = audio_text

    intent = agent_graph.tools["RecognizeIntent"].func(text)
    entities = agent_graph.tools["ExtractEntities"].func(text)
    consent = agent_graph.tools["CheckConsent"].func()

    if consent and intent == "book":
        ride = agent_graph.tools["BookRide"].func(
            entities["pickup"], entities["drop"], entities["payment_mode"]
        )
        agent_graph.tools["NotifyUser"].func(
            f"Ride booked for {user_id}: driver {ride['driver']}, ETA {ride['eta']} mins"
        )
        print(f"[Simulation result] Ride booked: {ride}")
    else:
        agent_graph.tools["NotifyUser"].func(f"Cannot book ride for {user_id}, consent missing")
        print("[Simulation result] Booking skipped due to missing consent")

if __name__ == "__main__":
    # Simulate with audio file
    asyncio.run(run_simulation("voice_input.wav", "user123"))
    # Simulate without audio file
    asyncio.run(run_simulation("", "user456"))
