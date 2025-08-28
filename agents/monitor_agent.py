import asyncio
from tools import notify_user

async def monitor_ride(ride: dict, user_id: str):
    """
    Simulated ride monitoring
    """
    for minute in range(1, 6):  # simulate 5 monitoring cycles
        await asyncio.sleep(2)  # short interval for demo
        # simulate off-route after 3 cycles
        location_status = "on-route" if minute < 3 else "off-route"
        print(f"[SIMULATION] Ride {ride['ride_id']} is {location_status}")
        if location_status == "off-route":
            notify_user(user_id, f"Alert: Ride {ride['ride_id']} off-route!")
