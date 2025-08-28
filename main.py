# main.py
import asyncio
from agents.orchestrator import agent_graph

async def run_simulation(audio_path: str = None, user_id: str = "user123"):
    if not audio_path:
        print("No audio provided. Please type your request instead.")
        user_input = "Simulated typed request"
    else:
        user_input = f"Simulated audio file: {audio_path}"

    response = await agent_graph.ainvoke({"input": user_input})
    print(response)

if __name__ == "__main__":
    # Run simulation with audio
    asyncio.run(run_simulation("voice_input.wav"))
    # Run simulation without audio
    asyncio.run(run_simulation(None))
