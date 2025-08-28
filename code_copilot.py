

# config.py
from typing import List, Optional
from pydantic import BaseSettings, AnyHttpUrl, Field

class Settings(BaseSettings):
    # Telegram
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_WEBHOOK_SECRET: str = Field(..., description="HMAC or shared secret for webhook authentication")

    # Azure Speech (ASR + optional TTS)
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None

    # Translation (Azure or Google)
    AZURE_TRANSLATOR_KEY: Optional[str] = None
    AZURE_TRANSLATOR_ENDPOINT: Optional[AnyHttpUrl] = None
    GOOGLE_TRANSLATE_KEY: Optional[str] = None

    # NLU (OpenAI or Rasa)
    OPENAI_API_KEY: Optional[str] = None
    RASA_URL: Optional[AnyHttpUrl] = None

    # Uber
    UBER_CLIENT_ID: str
    UBER_CLIENT_SECRET: str
    UBER_BASE_URL: AnyHttpUrl = "https://api.uber.com"  # or sandbox
    UBER_REDIRECT_URI: Optional[AnyHttpUrl] = None

    # Maps/Safety
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    # Notifications
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_FROM_NUMBER: Optional[str] = None

    # Privacy/consent
    CONSENT_TTL_SECONDS: int = 600
    DATA_RETENTION_SECONDS: int = 3600

    # Defaults
    DEFAULT_PAYMENT_MODE: str = "cash"
    EMERGENCY_CONTACTS: List[str] = []

    class Config:
        env_file = ".env"

settings = Settings()
python
# schemas.py
from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field

Language = Literal["te", "hi", "en"]

class DetectedLanguage(BaseModel):
    code: Language
    confidence: float

class Intent(BaseModel):
    label: Literal["book", "cancel", "schedule", "unknown"]
    confidence: float

class Entities(BaseModel):
    pickup: Optional[str] = None
    dropoff: Optional[str] = None
    when: Optional[str] = None  # ISO or natural language
    payment_mode: Optional[Literal["cash","card"]] = "cash"

class BookingRequest(BaseModel):
    user_id: str
    chat_id: str
    lang: Language
    text_original: str
    text_en: str
    intent: Intent
    entities: Entities

class BookingResult(BaseModel):
    request_id: str
    ride_id: Optional[str] = None
    status: Literal["pending","confirmed","failed"] = "pending"
    message: Optional[str] = None
    driver: Optional[Dict] = None
    eta_minutes: Optional[int] = None

class RideStatus(BaseModel):
    ride_id: str
    status: Literal["requested","accepted","arriving","in_progress","completed","canceled"]
    driver: Optional[Dict] = None
    eta_minutes: Optional[int] = None
    location: Optional[Dict] = None

class ConsentState(BaseModel):
    translation: bool = False
    booking: bool = False
    sharing: bool = False

class SafetyEvent(BaseModel):
    ride_id: str
    type: Literal["off_route","long_stop","panic"]
    details: Dict
Core services
python
# services/asr.py
import asyncio
from typing import Optional

class ASRService:
    def __init__(self, azure_key: Optional[str], azure_region: Optional[str]):
        self._azure_key = azure_key
        self._azure_region = azure_region

    async def transcribe_te(self, audio_bytes: bytes) -> str:
        # Minimal stub to keep the module dependency-light.
        # Replace with azure.cognitiveservices.speech SDK usage.
        # Tip: run blocking SDK calls via asyncio.to_thread(...)
        text = await asyncio.to_thread(self._azure_stub, audio_bytes)
        return text

    def _azure_stub(self, audio_bytes: bytes) -> str:
        # TODO: Implement Azure Speech SDK streaming recognition for Telugu
        raise NotImplementedError("Integrate Azure Speech SDK for Telugu ASR")
python
# services/lang_detect.py
from langdetect import detect_langs
from schemas import DetectedLanguage

class LangDetectService:
    def detect(self, text: str) -> DetectedLanguage:
        # langdetect returns list like 'te:0.99'
        langs = detect_langs(text)
        top = max(langs, key=lambda l: l.prob)
        code = top.lang if top.lang in {"te","hi","en"} else "en"
        return DetectedLanguage(code=code, confidence=float(top.prob))
python
# services/translate.py
import httpx
from typing import Optional

class TranslatorService:
    def __init__(self, azure_key: Optional[str], azure_endpoint: Optional[str], google_key: Optional[str]):
        self._azure_key = azure_key
        self._azure_endpoint = azure_endpoint
        self._google_key = google_key

    async def te_to_en(self, text: str) -> str:
        if self._azure_key and self._azure_endpoint:
            return await self._azure_translate(text, "te", "en")
        elif self._google_key:
            return await self._google_translate(text, "te", "en")
        return text  # fallback no-op

    async def _azure_translate(self, text: str, src: str, tgt: str) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            url = f"{self._azure_endpoint}/translate?api-version=3.0&from={src}&to={tgt}"
            headers = {"Ocp-Apim-Subscription-Key": self._azure_key}
            resp = await client.post(url, headers=headers, json=[{"text": text}])
            resp.raise_for_status()
            data = resp.json()
            return data[0]["translations"][0]["text"]

    async def _google_translate(self, text: str, src: str, tgt: str) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {"key": self._google_key, "q": text, "source": src, "target": tgt, "format": "text"}
            resp = await client.post(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data["data"]["translations"][0]["translatedText"]
python
# services/nlu.py
import json
from typing import Tuple
from schemas import Intent, Entities
import httpx
import os

class NLUService:
    def __init__(self, openai_api_key: str | None, rasa_url: str | None):
        self._openai_key = openai_api_key
        self._rasa_url = rasa_url

    async def classify_intent(self, text_en: str) -> Intent:
        if self._rasa_url:
            return await self._rasa_intent(text_en)
        return await self._openai_intent(text_en)

    async def extract_entities(self, text_en: str) -> Entities:
        if self._rasa_url:
            return await self._rasa_entities(text_en)
        return await self._openai_entities(text_en)

    async def _openai_intent(self, text: str) -> Intent:
        assert self._openai_key, "OPENAI_API_KEY required"
        schema = {"type":"object","properties":{"label":{"enum":["book","cancel","schedule","unknown"]},"confidence":{"type":"number"}}, "required":["label","confidence"]}
        msg = [
            {"role":"system","content":"Classify user intent among book, cancel, schedule, unknown."},
            {"role":"user","content":text},
        ]
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._openai_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": msg,
                    "response_format": {"type": "json_schema", "json_schema":{"name":"intent","schema":schema}}
                },
            )
            resp.raise_for_status()
            data = resp.json()
            parsed = json.loads(data["choices"][0]["message"]["content"])
            return Intent(**parsed)

    async def _openai_entities(self, text: str) -> Entities:
        assert self._openai_key, "OPENAI_API_KEY required"
        schema = {
            "type":"object",
            "properties":{
                "pickup":{"type":["string","null"]},
                "dropoff":{"type":["string","null"]},
                "when":{"type":["string","null"]},
                "payment_mode":{"enum":["cash","card"],"default":"cash"}
            },
            "required":["pickup","dropoff"]
        }
        msg = [
            {"role":"system","content":"Extract pickup, dropoff, when, payment_mode from the text. Be concise."},
            {"role":"user","content":text},
        ]
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._openai_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": msg,
                    "response_format": {"type":"json_schema","json_schema":{"name":"entities","schema":schema}}
                },
            )
            resp.raise_for_status()
            data = resp.json()
            parsed = json.loads(data["choices"][0]["message"]["content"])
            return Entities(**parsed)

    async def _rasa_intent(self, text: str) -> Intent:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{self._rasa_url}/model/parse", json={"text": text})
            resp.raise_for_status()
            d = resp.json()
            intent = d.get("intent", {})
            return Intent(label=intent.get("name","unknown"), confidence=float(intent.get("confidence",0.0)))

    async def _rasa_entities(self, text: str) -> Entities:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{self._rasa_url}/model/parse", json={"text": text})
            resp.raise_for_status()
            d = resp.json()
            ents = {e["entity"]: e["value"] for e in d.get("entities", []) if e.get("entity") in {"pickup","dropoff","when","payment_mode"}}
            return Entities(**ents)
Integrations (Uber, Maps/Safety, Notifications, Consent)
python
# services/uber.py
from typing import Optional, Dict
import httpx
import time

class UberClient:
    def __init__(self, client_id: str, client_secret: str, base_url: str):
        self._id = client_id
        self._secret = client_secret
        self._base = base_url
        self._token: Optional[str] = None
        self._exp = 0

    async def _auth(self):
        if self._token and time.time() < self._exp - 60:
            return
        # TODO: Implement OAuth flows appropriate for your Uber integration
        raise NotImplementedError("Implement Uber OAuth and token storage")

    async def book_cash_ride(self, pickup: str, dropoff: str) -> Dict:
        await self._auth()
        # TODO: Geocode pickup/dropoff, validate service availability
        # Example payload structure; adjust to Uber API spec
        payload = {
            "pickup": {"address": pickup},
            "dropoff": {"address": dropoff},
            "payment_method": "cash"
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(f"{self._base}/v2/requests", headers={"Authorization": f"Bearer {self._token}"}, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def ride_status(self, ride_id: str) -> Dict:
        await self._auth()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base}/v2/requests/{ride_id}", headers={"Authorization": f"Bearer {self._token}"})
            resp.raise_for_status()
            return resp.json()
python
# services/maps.py
from typing import Dict, List, Tuple, Optional
import httpx
import math

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

class SafetyService:
    def __init__(self, google_key: Optional[str]):
        self._key = google_key

    async def expected_route(self, origin: str, dest: str) -> List[Tuple[float,float]]:
        # TODO: Call Directions API and decode polyline
        return []

    def off_route(self, current: Tuple[float,float], route: List[Tuple[float,float]], threshold_m: float = 200.0) -> bool:
        # Simple nearest-point distance to route vertices
        if not route:
            return False
        dmin = min(haversine_m(current[0],current[1],p[0],p[1]) for p in route)
        return dmin > threshold_m
python
# services/notify.py
import httpx
from typing import Optional, List

class TelegramService:
    def __init__(self, bot_token: str):
        self._base = f"https://api.telegram.org/bot{bot_token}"

    async def send_message(self, chat_id: str, text: str, reply_to: Optional[int] = None):
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{self._base}/sendMessage", json={"chat_id": chat_id, "text": text, "reply_to_message_id": reply_to})

    async def send_location(self, chat_id: str, lat: float, lon: float):
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{self._base}/sendLocation", json={"chat_id": chat_id, "latitude": lat, "longitude": lon})

    async def get_file_url(self, file_id: str) -> str:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{self._base}/getFile", params={"file_id": file_id})
            r.raise_for_status()
            file_path = r.json()["result"]["file_path"]
            return f"https://api.telegram.org/file/bot{self._base.split('bot')[1]}/{file_path}"

class TwilioService:
    def __init__(self, sid: Optional[str], token: Optional[str], from_number: Optional[str]):
        self._sid = sid; self._token = token; self._from = from_number

    async def send_sms(self, to: str, text: str):
        if not (self._sid and self._token and self._from):
            return
        async with httpx.AsyncClient(auth=(self._sid, self._token), timeout=10) as client:
            await client.post(f"https://api.twilio.com/2010-04-01/Accounts/{self._sid}/Messages.json",
                              data={"To": to, "From": self._from, "Body": text})
python
# services/consent.py
import time
from typing import Dict
from schemas import ConsentState

class ConsentService:
    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._store: Dict[str, tuple[ConsentState, float]] = {}

    def get(self, user_id: str) -> ConsentState:
        state, exp = self._store.get(user_id, (ConsentState(), 0))
        if time.time() > exp:
            return ConsentState()
        return state

    def set(self, user_id: str, state: ConsentState):
        self._store[user_id] = (state, time.time() + self._ttl)
Orchestrator
python
# orchestrator.py
import asyncio
import uuid
from typing import Optional
from schemas import BookingRequest, BookingResult, RideStatus, ConsentState
from services.asr import ASRService
from services.lang_detect import LangDetectService
from services.translate import TranslatorService
from services.nlu import NLUService
from services.uber import UberClient
from services.maps import SafetyService
from services.notify import TelegramService, TwilioService
from services.consent import ConsentService
from config import settings

class BookingOrchestrator:
    def __init__(self,
                 asr: ASRService,
                 langdet: LangDetectService,
                 translator: TranslatorService,
                 nlu: NLUService,
                 uber: UberClient,
                 safety: SafetyService,
                 tg: TelegramService,
                 twilio: TwilioService,
                 consent: ConsentService):
        self.asr = asr; self.langdet = langdet; self.translator = translator
        self.nlu = nlu; self.uber = uber; self.safety = safety
        self.tg = tg; self.twilio = twilio; self.consent = consent

    async def handle_voice_booking(self, chat_id: str, user_id: str, audio_bytes: bytes) -> BookingResult:
        # 1) ASR
        text_te = await self.asr.transcribe_te(audio_bytes)

        # 2) Language detection
        lang = self.langdet.detect(text_te)

        # 3) Consent: translation
        consent = self.consent.get(user_id)
        if not consent.translation:
            await self.tg.send_message(chat_id, "అనువాదానికి సమ్మతి ఇస్తారా? (Do you consent to translation?) Reply: YES/NO")
            return BookingResult(request_id=str(uuid.uuid4()), status="pending", message="Awaiting translation consent")

        # 4) Translate to English
        text_en = await self.translator.te_to_en(text_te) if lang.code == "te" else text_te

        # 5) Intent + Entities
        intent = await self.nlu.classify_intent(text_en)
        entities = await self.nlu.extract_entities(text_en)

        req = BookingRequest(
            user_id=user_id, chat_id=chat_id, lang=lang.code, text_original=text_te, text_en=text_en,
            intent=intent, entities=entities
        )

        if intent.label != "book":
            await self.tg.send_message(chat_id, "ఈ అభ్యర్థన బుకింగ్ కాదు. (Not a booking request.)")
            return BookingResult(request_id=str(uuid.uuid4()), status="failed", message="Not a booking intent")

        # 6) Confirm details in Telugu before booking (privacy + correctness)
        confirm_msg = f"దయచేసి నిర్ధారించండి: పికప్ '{entities.pickup}', డ్రాప్ '{entities.dropoff}', చెల్లింపు '{entities.payment_mode}'. (Confirm: pickup, dropoff, payment.) Reply YES/NO"
        await self.tg.send_message(chat_id, confirm_msg)
        return BookingResult(request_id=str(uuid.uuid4()), status="pending", message="Awaiting booking confirmation")

    async def finalize_booking(self, chat_id: str, user_id: str, confirmed: bool, last_entities) -> BookingResult:
        if not confirmed:
            await self.tg.send_message(chat_id, "రద్దు చేయబడింది. (Canceled.)")
            return BookingResult(request_id=str(uuid.uuid4()), status="failed", message="User declined")

        # Booking consent
        consent = self.consent.get(user_id)
        if not consent.booking:
            await self.tg.send_message(chat_id, "బుకింగ్‌కు సమ్మతి ఇస్తారా? (Consent to book?) Reply YES/NO")
            return BookingResult(request_id=str(uuid.uuid4()), status="pending", message="Awaiting booking consent")

        # 7) Book ride (cash)
        res = await self.uber.book_cash_ride(pickup=last_entities.pickup, dropoff=last_entities.dropoff)
        ride_id = res.get("request_id") or res.get("ride_id")

        # 8) Notify user with driver details
        driver = res.get("driver", {})
        eta = res.get("eta", {}).get("minutes")
        await self.tg.send_message(chat_id, f"బుకింగ్ పూర్తయ్యింది. డ్రైవర్: {driver.get('name','?')}, ETA: {eta} నిమి. (Booking confirmed.)")

        # 9) Start monitors
        asyncio.create_task(self._monitor_eta_and_status(chat_id, ride_id))
        asyncio.create_task(self._monitor_safety(chat_id, ride_id, last_entities.pickup, last_entities.dropoff))

        return BookingResult(request_id=str(uuid.uuid4()), ride_id=ride_id, status="confirmed", driver=driver, eta_minutes=eta)

    async def _monitor_eta_and_status(self, chat_id: str, ride_id: str):
        while True:
            try:
                st = await self.uber.ride_status(ride_id)
                status = st.get("status","")
                eta = st.get("eta",{}).get("minutes")
                if status in {"arriving","accepted"} and eta is not None:
                    await self.tg.send_message(chat_id, f"ETA: {eta} నిమిషాలు.")
                if status in {"in_progress","completed","canceled"}:
                    await self.tg.send_message(chat_id, f"స్థితి: {status}")
                    if status != "in_progress":
                        break
            except Exception:
                pass
            await asyncio.sleep(20)

    async def _monitor_safety(self, chat_id: str, ride_id: str, pickup: str, dropoff: str):
        route = await self.safety.expected_route(pickup, dropoff)
        off_count = 0
        while True:
            try:
                st = await self.uber.ride_status(ride_id)
                status = st.get("status","")
                if status in {"completed","canceled"}:
                    break
                loc = st.get("location") or {}
                lat, lon = loc.get("lat"), loc.get("lng")
                if lat and lon and self.safety.off_route((lat,lon), route):
                    off_count += 1
                    if off_count >= 3:
                        await self.tg.send_message(chat_id, "హెచ్చరిక: మార్గం నుండి వ్యత్యయం గుర్తించబడింది. (Off-route detected.)")
                        # Emergency contacts via SMS
                        for to in settings.EMERGENCY_CONTACTS:
                            await self.twilio.send_sms(to, f"Off-route detected for ride {ride_id}")
                        off_count = 0
            except Exception:
                pass
            await asyncio.sleep(30)
FastAPI webhook and consent handling
python
# app.py
import base64
import hmac, hashlib, json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from config import settings
from services.asr import ASRService
from services.lang_detect import LangDetectService
from services.translate import TranslatorService
from services.nlu import NLUService
from services.uber import UberClient
from services.maps import SafetyService
from services.notify import TelegramService, TwilioService
from services.consent import ConsentService
from orchestrator import BookingOrchestrator
from schemas import ConsentState, Entities

app = FastAPI()

# Instantiate services
asr = ASRService(settings.AZURE_SPEECH_KEY, settings.AZURE_SPEECH_REGION)
langdet = LangDetectService()
translator = TranslatorService(settings.AZURE_TRANSLATOR_KEY, str(settings.AZURE_TRANSLATOR_ENDPOINT) if settings.AZURE_TRANSLATOR_ENDPOINT else None, settings.GOOGLE_TRANSLATE_KEY)
nlu = NLUService(settings.OPENAI_API_KEY, str(settings.RASA_URL) if settings.RASA_URL else None)
uber = UberClient(settings.UBER_CLIENT_ID, settings.UBER_CLIENT_SECRET, str(settings.UBER_BASE_URL))
safety = SafetyService(settings.GOOGLE_MAPS_API_KEY)
tg = TelegramService(settings.TELEGRAM_BOT_TOKEN)
twilio = TwilioService(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.TWILIO_FROM_NUMBER)
consent = ConsentService(settings.CONSENT_TTL_SECONDS)
orch = BookingOrchestrator(asr, langdet, translator, nlu, uber, safety, tg, twilio, consent)

def verify_webhook(req: Request, body: bytes):
    # Optional: implement shared-secret/HMAC verification for Telegram reverse proxy
    sig = req.headers.get("X-TG-Secret", "")
    if not hmac.compare_digest(sig, settings.TELEGRAM_WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    body = await request.body()
    # If you're putting this behind a proxy that adds a secret header, uncomment verify.
    # verify_webhook(request, body)
    update = json.loads(body)
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat", {})
    chat_id = str(chat.get("id"))
    from_user = msg.get("from", {})
    user_id = str(from_user.get("id"))

    # Consent responses
    text = msg.get("text", "") or ""
    if text.strip().upper() in {"YES","NO"}:
        state = consent.get(user_id)
        # Progressively collect consents in order: translation -> booking -> sharing
        if not state.translation and text.strip().upper() == "YES":
            state.translation = True; consent.set(user_id, state)
            await tg.send_message(chat_id, "ధన్యవాదాలు. కొనసాగిద్దాం. (Thanks, continuing.)")
        elif state.translation and not state.booking and text.strip().upper() == "YES":
            state.booking = True; consent.set(user_id, state)
            await tg.send_message(chat_id, "బుకింగ్ సమ్మతి నమోదు. (Booking consent recorded.)")
        elif state.booking and not state.sharing and text.strip().upper() == "YES":
            state.sharing = True; consent.set(user_id, state)
            await tg.send_message(chat_id, "షేరింగ్ సమ్మతి నమోదు. (Sharing consent recorded.)")
        else:
            await tg.send_message(chat_id, "సమ్మతి నవీకరించబడింది. (Consent updated.)")
        return JSONResponse({"ok": True})

    # Voice flow
    voice = msg.get("voice")
    if voice:
        file_id = voice.get("file_id")
        file_url = await tg.get_file_url(file_id)
        # Download audio
        async with httpx.AsyncClient(timeout=20) as client:
            audio = await client.get(file_url)
            audio.raise_for_status()
            audio_bytes = audio.content
        res = await orch.handle_voice_booking(chat_id, user_id, audio_bytes)
        return JSONResponse(res.dict())

    # Text fallback (English or Telugu typed)
    if text:
        # Minimal text flow for quick testing
        lang = langdet.detect(text)
        text_en = await translator.te_to_en(text) if lang.code == "te" else text
        intent = await nlu.classify_intent(text_en)
        entities = await nlu.extract_entities(text_en)
        if intent.label == "book":
            await tg.send_message(chat_id, f"Confirm booking? pickup='{entities.pickup}', dropoff='{entities.dropoff}', payment='{entities.payment_mode}'. Reply YES/NO")
            return JSONResponse({"ok": True})
        await tg.send_message(chat_id, "Not a booking request.")
        return JSONResponse({"ok": True})

    return JSONResponse({"ok": True})

@app.post("/consent/{user_id}")
async def set_consent(user_id: str, state: ConsentState):
    consent.set(user_id, state)
    return {"ok": True, "state": state.dict()}

@app.get("/healthz")
async def health():
    return {"ok": True}
Notes
Replace NotImplementedError stubs with concrete SDK calls (Azure Speech SDK, Uber OAuth, Google Directions polyline decode).

For persistence, back the ConsentService and monitors with a datastore and job queue (Redis, Postgres, Celery).

To play confirmations in Telugu, add Azure Speech Synthesis in a small TTS service and send as Telegram voice note.

If you want, I can fill in any specific provider implementation next (e.g., Azure Speech SDK code, Uber OAuth + booking flows, or Google Directions + polyline decode), and wire tests around each module.