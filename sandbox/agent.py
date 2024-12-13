from __future__ import annotations

import logging
import pandas as pd
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def generate_report(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    
    switches = df[df["Hostname"].str.contains("SWP1-STACK", na=False)].copy()
    reports = []

    for i, row in switches.iterrows():
        modelo = row["PID"]
        cpu = row["CPU Average"]
        
        power_supplies = df[df["Descriptivo"].str.contains(f"Switch {i + 1} - Power Supply", na=False)]
        power_info = [f"{ps['Descriptivo']} - {ps['Power Health']}" for _, ps in power_supplies.iterrows()]

        report = (
            f"Switch {i + 1}:\n"
            f"Modelo: {modelo}\n"
            f"CPU: {cpu}\n"
            f"Módulos de Alimentación: {len(power_supplies)} ({', '.join(power_info)})"
        )

        reports.append(report)

    return "\n\n".join(reports)


def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=(
            "Saluda diciendo tu nombre el cual es SOPHIA, luego presentas quién eres y di 'Bienvenido al Experience Operation Center'. "
            "Eres la Inteligencia Artificial de la empresa TXDX SECURE. "
            "Tu función es atender a los clientes que hagan una llamada para monitorear sus equipos y resolver dudas."
            "TXDX SECURE es una empresa dedicada a redes y ciberseguridad. "
            "Te harán preguntas de ciberseguridad, así que prepárate para eso. Si hay algún dato sobre la empresa que no sepas, invéntalo."
        ),
        modalities=["audio", "text"],
        model="gpt-4o-realtime-preview-2024-10-01",
        voice="sage"
    )
    assistant = MultimodalAgent(model=model)
    assistant.start(ctx.room, participant)

    # Generar informe
    report = generate_report("output.csv")

    # Crear el mensaje con el informe
    model.sessions[0].conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content=f"Este es el informe solicitado:\n\n{report}",
        )
    )
    model.sessions[0].response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
