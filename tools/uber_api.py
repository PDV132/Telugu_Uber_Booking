import asyncio

async def book_ride(pickup: str, drop: str, payment_mode: str) -> dict:
    """
    Simulated Uber booking.
    """
    print(f"[SIMULATION] Booking ride from {pickup} to {drop} with payment {payment_mode}")
    return {
        "ride_id": "SIM123",
        "driver": "Simulated Driver",
        "eta": 5,
        "pickup": pickup,
        "drop": drop,
        "payment_mode": payment_mode
    }
