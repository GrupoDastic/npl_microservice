from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
import torch
import json
import os
from db.db import get_pg_connection
import re  # Global import added here

load_dotenv()

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "..", "model")

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
except Exception as e:
    raise e

config_path = os.path.join(model_path, "config.json")
try:
    with open(config_path) as f:
        config = json.load(f)
        id2label = config["id2label"]
        label2id = config["label2id"]
except Exception as e:
    raise e

app = FastAPI()


class Query(BaseModel):
    text: str


def execute_query(query, params=()):
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.post("/predict")
def predict(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
    command = id2label[str(prediction)]
    print(f"Command recognized: {command}")

    if command == "cm1":
        rows = execute_query("""
                             SELECT z.identifier, s.strip_identifier, ps.identifier
                             FROM parking_spaces ps
                                      JOIN strips s
                                           ON s.zone_id = ps.zone_id AND s.strip_identifier = ps.strip_identifier
                                      JOIN zones z ON z.id = ps.zone_id
                             WHERE ps.status = 'free'
                             ORDER BY CASE z.identifier
                                          WHEN 'B' THEN 1
                                          WHEN 'D' THEN 2
                                          WHEN 'C' THEN 3
                                          WHEN 'H' THEN 4
                                          WHEN 'E' THEN 5
                                          WHEN 'G' THEN 6
                                          ELSE 7
                                          END,
                                      ps.last_updated DESC
                             LIMIT 3;
                             """)
        if not rows:
            return {"respuesta": "Lo siento, no hay parqueaderos disponibles en este momento."}
        zonas = {}
        for zona, franja, parqueadero in rows:
            key = f"zona {zona}, franja {franja}"
            zonas.setdefault(key, []).append(parqueadero)
        frases = [f"{ubicacion}: {', '.join(parqueos)}" for ubicacion, parqueos in zonas.items()]
        return {"respuesta": "Existen parqueaderos disponibles en:\n" + "\n".join(frases)}

    if command == "cm2":
        rows = execute_query("""
            SELECT DISTINCT z.identifier
            FROM parking_spaces ps
            JOIN zones z ON z.id = ps.zone_id
            WHERE ps.status = 'free';
        """)
        zonas = [r[0] for r in rows]
        if zonas:
            return {"respuesta": "Zonas con parqueaderos disponibles: " + ", ".join(zonas)}
        return {"respuesta": "No hay zonas con parqueaderos disponibles actualmente."}

    if command == "cm3":
        return {"respuesta": "Esta función aún no está implementada para repetir la última información."}

    if command == "cm4":
        match = re.search(r"(\w+\d+-?\d*)", query.text)
        if not match:
            return {"respuesta": "Por favor, indique un identificador de parqueadero válido."}
        parqueadero_id = match.group(1)
        rows = execute_query("""
            SELECT ps.status, z.identifier, s.strip_identifier
            FROM parking_spaces ps
            JOIN strips s ON ps.zone_id = s.zone_id AND ps.strip_identifier = s.strip_identifier
            JOIN zones z ON z.id = ps.zone_id
            WHERE ps.identifier = %s;
        """, (parqueadero_id,))
        if not rows:
            return {"respuesta": f"El parqueadero {parqueadero_id} no existe."}
        status, zona, franja = rows[0]
        disponible = "disponible" if status == "free" else "ocupado"
        return {"respuesta": f"El parqueadero {parqueadero_id} en zona {zona} franja {franja} está {disponible}."}

    if command == "cm5":
        zonas_posibles = ["B", "C", "D", "G", "H"]
        zona_detectada = next((z for z in zonas_posibles if z.lower() in query.text.lower()), None)
        if not zona_detectada:
            return {"respuesta": "Por favor, indique una zona válida (ej: zona B, zona C)."}
        rows = execute_query("""
            SELECT z.name, s.strip_identifier, ps.identifier
            FROM parking_spaces ps
            JOIN strips s ON s.zone_id = ps.zone_id AND s.strip_identifier = ps.strip_identifier
            JOIN zones z ON z.id = ps.zone_id
            WHERE ps.status = 'free' AND z.identifier = %s
            LIMIT 5;
        """, (zona_detectada.upper(),))
        if not rows:
            return {"respuesta": f"No hay parqueaderos disponibles en la zona {zona_detectada}."}
        zonas = {}
        for zona, franja, parqueadero in rows:
            key = f"zona {zona}, franja {franja}"
            zonas.setdefault(key, []).append(parqueadero)
        frases = [f"{ubicacion}: {', '.join(parqueos)}" for ubicacion, parqueos in zonas.items()]
        return {"respuesta": f"Parqueaderos disponibles en zona {zona_detectada.upper()}:\n" + "\n".join(frases)}

    if command == "cm6":
        match = re.search(r"(\w+\d+-?\d*)", query.text)
        if not match:
            return {"respuesta": "Por favor, indique un parqueadero válido (ej: G1-03)."}
        parqueadero_id = match.group(1)
        rows = execute_query("""
            SELECT z.name, z.identifier, s.strip_identifier, ps.status
            FROM parking_spaces ps
            JOIN strips s ON s.zone_id = ps.zone_id AND s.strip_identifier = ps.strip_identifier
            JOIN zones z ON z.id = ps.zone_id
            WHERE ps.identifier = %s;
        """, (parqueadero_id,))
        if not rows:
            return {"respuesta": f"El parqueadero {parqueadero_id} no existe."}
        zona_name, zona_id, franja, status = rows[0]
        estado = "disponible" if status == "free" else "ocupado"
        return {
            "respuesta": f"El parqueadero {parqueadero_id} pertenece a la zona {zona_name} ({zona_id}) en la franja {franja} y está {estado}."}

    if command == "cm7":
        return {"respuesta": "Hola 👋, puedo ayudarte a encontrar parqueaderos disponibles. Solo dime lo que necesitas."}

    if command == "cm8":
        return {"respuesta": "Puedo ayudarte con:\n"
                             "- Buscar parqueaderos disponibles en general (cm1)\n"
                             "- Ver qué zonas tienen parqueo libre (cm2)\n"
                             "- Ver parqueos disponibles en una zona (cm5)\n"
                             "- Saber si un parqueo específico está libre (cm4)\n"
                             "- Saber a qué zona pertenece un parqueo (cm6)"}

        if command == "cm9":
            return {"respuesta": "Lo siento, no entendí tu mensaje. ¿Podrías reformularlo o pedirme ayuda?"}

    if command == "cm10":
        zonas_posibles = ["B", "C", "D", "G", "H"]
        zona_detectada = next((z for z in zonas_posibles if z.lower() in query.text.lower()), None)
        if not zona_detectada:
            return {"respuesta": "Por favor, indique una zona válida para listar sus franjas."}
        rows = execute_query("""
            SELECT DISTINCT s.strip_identifier
            FROM strips s
            JOIN zones z ON z.id = s.zone_id
            WHERE z.identifier = %s;
        """, (zona_detectada.upper(),))
        if not rows:
            return {"respuesta": f"No se encontraron franjas para la zona {zona_detectada}."}
        franjas = [str(f[0]) for f in rows]
        return {"respuesta": f"La zona {zona_detectada} tiene las siguientes franjas: " + ", ".join(franjas)}

    if command == "cm11":
        zona_match = re.search(r"zona\s+([A-Z])", query.text, re.IGNORECASE)
        franja_match = re.search(r"franja\s+(\d+)", query.text, re.IGNORECASE)


        if not zona_match or not franja_match:
            return {"respuesta": "Por favor, indique una zona y franja válidas (ej: zona B franja 1)."}

        zona = zona_match.group(1).upper()
        franja = int(franja_match.group(1))

        rows = execute_query("""
        SELECT ps.identifier
        FROM parking_spaces ps
        JOIN zones z ON z.id = ps.zone_id
        WHERE z.identifier = %s AND ps.strip_identifier = %s AND ps.status = 'free';
        """, (zona, franja))

        if not rows:
            return {"respuesta": f"No hay parqueaderos disponibles en la zona {zona}, franja {franja}."}

        parqueaderos = [r[0] for r in rows]
        return {
            "respuesta": f"En la zona {zona}, franja {franja}, los siguientes parqueaderos están disponibles:\\n- " + "\\n- ".join(
                parqueaderos)
        }

    return {"respuesta": f"Comando reconocido: {command}"}
