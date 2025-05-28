#!/usr/bin/env python3
# siraj_pipeline.py

import os
import sys
import asyncio
import re
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Pipecat core
from pipecat.pipeline.pipeline    import Pipeline
from pipecat.pipeline.task        import PipelineTask, PipelineParams
from pipecat.pipeline.runner      import PipelineRunner
from pipecat_flows    import FlowConfig, FlowManager, FlowResult, FlowArgs

# Pipecat transports & processors
from pipecat.transports.services.daily          import DailyTransport, DailyParams
from pipecat.services.azure                     import AzureSTTService
from pipecat.services.openai                    import OpenAILLMService
from pipecat.services.elevenlabs                import ElevenLabsTTSService
from pipecat.audio.vad.silero                   import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer             import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.transcript_processor    import TranscriptProcessor
from pipecat.processors.frameworks.rtvi         import (
    RTVISpeakingProcessor, RTVIUserTranscriptionProcessor,
    RTVIBotLLMProcessor, RTVIBotTTSProcessor,
    RTVIBotTranscriptionProcessor, RTVIMetricsProcessor,
    FrameDirection,
)
from pipecat.frames.frames                      import EndFrame

load_dotenv(override=True)
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# ─── Load your CSV of restaurants + paths ───────────────────────────────────
CSV_PATH = os.getenv("CSV_PATH", "restaurants2_neighborhood_stations_paths6.csv")
df = pd.read_csv(CSV_PATH)
df["Display_Name"] = df["Name"].fillna(df.get("English_name",""))

def lookup_route(restaurant_name: str) -> str:
    """Simple exact/fuzzy lookup; returns the 'Path' column or empty."""
    # exact
    hit = df[df["Display_Name"] == restaurant_name]
    if not hit.empty:
        return hit.iloc[0]["Path"]
    # fallback fuzzy
    from fuzzywuzzy import process
    names = df["Display_Name"].tolist()
    best, score = process.extractOne(restaurant_name, names)
    if score >= 60:
        return df[df["Display_Name"] == best].iloc[0]["Path"]
    return ""

# ─── Handlers for Pipecat Flows ──────────────────────────────────────────────
async def get_route_handler(args: FlowArgs) -> FlowResult:
    name = args["restaurant"]
    path = lookup_route(name)
    if not path:
        return FlowResult(output=f"عذراً، لا أجد توجيهات لمطعم “{name}”.")
    return FlowResult(output=f"أقرب مسار لمطعم {name} هو: {path}")

# ─── Build your flow_config ─────────────────────────────────────────────────
flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "role_messages": [
                {"role":"system", 
                 "content": """
أنت “سراج” مساعد محطة المترو. اسأل المستخدم “كيف يمكنني مساعدتك؟”
عندما يسمع سؤالاً عن مطعم، استدعي الدالة get_route.
                 """
                }
            ],
            "functions": [
                {
                    "type":"function",
                    "function":{
                        "name": "get_route",
                        "description": "احصل على توجيهات الوصول إلى مطعم عن طريق اسم المطعم",
                        "parameters":{
                            "type":"object",
                            "properties":{
                                "restaurant":{
                                    "type":"string",
                                    "description":"اسم المطعم بالعربية"
                                }
                            },
                            "required":["restaurant"]
                        },
                        "handler": get_route_handler,
                        "transition_to":"end"
                    }
                }
            ]
        },
        "end": {
            "task_messages":[
                {"role":"system",
                 "content":"إذا احتاج المستخدم إلى مساعدة إضافية فاطلب منه السؤال مرة أخرى، وإلا ودّعه."}
            ],
            "post_actions":[
                {"type":"tts_say", "text":"مع السلامة! أي سؤال آخر؟"},
                {"type":"end_conversation", "handler": lambda *_: None}
            ]
        }
    }
}

# ─── Wire up Pipecat services ────────────────────────────────────────────────
transport = DailyTransport(
    os.getenv("DAILY_ROOM_URL"),
    os.getenv("DAILY_ROOM_TOKEN"),
    "سراج – Metro Guide",
    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
        transcription_enabled=False,
        vad_enabled=True,
        vad_audio_passthrough=True,
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=100)
        )
    )
)

stt = AzureSTTService(
    api_key=os.getenv("AZURE_SPEECH_KEY"),
    region=os.getenv("AZURE_SPEECH_REGION"),
    language="ar-SA",
    sample_rate=24000,
    channels=1
)

tts = ElevenLabsTTSService(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="ar-SESalaamNeural",  # pick a good Arabic neural voice
    model="eleven_multilingual_v2",
    params=ElevenLabsTTSService.InputParams(stability=0.7, similarity_boost=0.7)
)

llm = OpenAILLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.2
)

# context & processors
context      = OpenAILLMContext()
aggregator   = llm.create_context_aggregator(context)
rtvi_speak   = RTVISpeakingProcessor()
rtvi_user    = RTVIUserTranscriptionProcessor()
rtvi_bot_llm = RTVIBotLLMProcessor()
rtvi_bot_tts = RTVIBotTTSProcessor(direction=FrameDirection.DOWNSTREAM)
rtvi_bot_tx  = RTVIBotTranscriptionProcessor()
rtvi_metrics = RTVIMetricsProcessor()
transcript   = TranscriptProcessor()

pipeline = Pipeline([
    transport.input(),
    rtvi_speak,
    stt,
    rtvi_user,
    transcript.user(),
    aggregator.user(),
    llm,
    rtvi_bot_llm,
    tts,
    rtvi_bot_tts,
    rtvi_bot_tx,
    transport.output(),
    transcript.assistant(),
    aggregator.assistant(),
    rtvi_metrics
])

# ─── Build & run ─────────────────────────────────────────────────────────────
async def main():
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    global _pipeline_task
    _pipeline_task = task

    flow = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=aggregator,
        tts=tts,
        flow_config=flow_config
    )
    await flow.initialize()

    @transport.event_handler("on_first_participant_joined")
    async def _(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
    
    await PipelineRunner().run(task)

if __name__ == "__main__":
    asyncio.run(main())
