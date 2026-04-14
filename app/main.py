"""
API FastAPI para clasificacion de intenciones - Parqueadero NLP
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
import json
import os
import re

try:
    from app.db.db import get_pg_connection
except ImportError:
    try:
        from db.db import get_pg_connection
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from db.db import get_pg_connection

load_dotenv()

CONFIDENCE_THRESHOLD = 0.7

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "..", "model")

model = None
tokenizer = None
id2label = None
label2id = None

last_results = []        # para evitar repetir parqueaderos
last_response_text = ""  # para cm3 (repetir respuesta)
MAX_HISTORY = 20

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    print(f"[NLP] Modelo cargado desde {model_path}")

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
        id2label = config["id2label"]
        label2id = config["label2id"]
    print(f"[NLP] Config cargada: {len(id2label)} clases")
except Exception as e:
    print(f"[NLP] ADVERTENCIA: No se pudo cargar el modelo: {e}")
    print(f"[NLP] La API iniciara pero /predict no funcionara hasta entrenar el modelo.")

app = FastAPI(title="Parqueadero NLP Bot")


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


def classify(text):
    if model is None or tokenizer is None:
        return "cm9", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    command = id2label[str(prediction.item())]
    conf = confidence.item()

    print(f"[NLP] Texto: '{text}' -> {command} (confianza: {conf:.3f})")

    if conf < CONFIDENCE_THRESHOLD:
        print(f"[NLP] Confianza baja ({conf:.3f} < {CONFIDENCE_THRESHOLD}), redirigiendo a cm9")
        return "cm9", conf

    return command, conf


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(query: Query):
    global last_response_text
    if model is None:
        return {"respuesta": "El modelo no esta cargado. Ejecuta primero: python training/train_model.py"}

    if not query.text or not query.text.strip():
        return {"respuesta": "Por favor, escribe algo para que pueda ayudarte."}

    command, confidence = classify(query.text)

    if command == "cm1":
        global last_results
        exclude_clause = ""
        params = ()

        if last_results:
            placeholders = ",".join(["%s"] * len(last_results))
            exclude_clause = f"AND ps.identifier NOT IN ({placeholders})"
            params = tuple(last_results)

        rows = execute_query(f"""
                SELECT z.identifier, s.strip_identifier, ps.identifier
                FROM parking_spaces ps
                JOIN strips s ON s.zone_id = ps.zone_id AND s.strip_identifier = ps.strip_identifier
                JOIN zones z ON z.id = ps.zone_id
                WHERE ps.status = 'free'
                {exclude_clause}
                ORDER BY RANDOM()
                LIMIT 3;
            """, params)

        # fallback
        if not rows:
            rows = execute_query("""
                                 SELECT z.identifier, s.strip_identifier, ps.identifier
                                 FROM parking_spaces ps
                                          JOIN strips s ON s.zone_id = ps.zone_id AND s.strip_identifier = ps.strip_identifier
                                          JOIN zones z ON z.id = ps.zone_id
                                 WHERE ps.status = 'free'
                                 ORDER BY RANDOM()
                                 LIMIT 3;
                                 """)

        if not rows:
            last_response_text = "No hay parqueaderos disponibles"
            return {"respuesta": last_response_text}

        zonas = {}
        for zona, franja, parqueadero in rows:
            key = f"zona {zona}, franja {franja}"
            zonas.setdefault(key, []).append(parqueadero)

        frases = [f"{k}: {', '.join(v)}" for k, v in zonas.items()]
        respuesta = "Parqueaderos disponibles:\n" + "\n".join(frases)

        # guardar memoria
        last_results = [r[2] for r in rows]
        last_response_text = respuesta

        return {"respuesta": respuesta}

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

        if not last_response_text:
            return {"respuesta": "Primero haz una consulta"}

        return {
            "respuesta": f"Te repito:\n\n{last_response_text}"
        }

    if command == "cm4":
        match = re.search(r"([A-Za-z]\d+-\d+)", query.text)
        if not match:
            match = re.search(r"(\w+\d+-?\d*)", query.text)
        if not match:
            return {"respuesta": "Por favor, indique un identificador de parqueadero valido (ej: G1-03)."}
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
        return {"respuesta": f"El parqueadero {parqueadero_id} en zona {zona} franja {franja} esta {disponible}."}

    if command == "cm5":
        zonas_posibles = ["B", "C", "D", "E", "G", "H"]
        zona_detectada = next((z for z in zonas_posibles if z.lower() in query.text.lower()), None)
        if not zona_detectada:
            return {"respuesta": "Por favor, indique una zona valida (ej: zona B, zona C)."}
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
        match = re.search(r"([A-Za-z]\d+-\d+)", query.text)
        if not match:
            match = re.search(r"(\w+\d+-?\d*)", query.text)
        if not match:
            return {"respuesta": "Por favor, indique un parqueadero valido (ej: G1-03)."}
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
            "respuesta": f"El parqueadero {parqueadero_id} pertenece a la zona {zona_name} ({zona_id}) en la franja {franja} y esta {estado}."
        }

    if command == "cm7":
        return {"respuesta": "Hola, puedo ayudarte a encontrar parqueaderos disponibles. Solo dime lo que necesitas."}

    if command == "cm8":
        return {
            "respuesta": (
                "Puedo ayudarte con:\n"
                "- Buscar parqueaderos disponibles en general\n"
                "- Ver que zonas tienen parqueo libre\n"
                "- Ver parqueos disponibles en una zona\n"
                "- Saber si un parqueo especifico esta libre\n"
                "- Saber a que zona pertenece un parqueo\n"
                "- Ver las franjas de una zona\n"
                "- Buscar parqueos por zona y franja"
            )
        }

    if command == "cm9":
        return {"respuesta": "Lo siento, no entendi tu mensaje. Podrias reformularlo o pedirme ayuda?"}

    if command == "cm10":
        zonas_posibles = ["B", "C", "D", "E", "G", "H"]
        zona_detectada = next((z for z in zonas_posibles if z.lower() in query.text.lower()), None)
        if not zona_detectada:
            return {"respuesta": "Por favor, indique una zona valida para listar sus franjas."}
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
        zona_match = re.search(r"zona\s+([A-Za-z])", query.text, re.IGNORECASE)
        franja_match = re.search(r"franja\s+(\d+)", query.text, re.IGNORECASE)

        if not zona_match or not franja_match:
            return {"respuesta": "Por favor, indique una zona y franja validas (ej: zona B franja 1)."}

        zona = zona_match.group(1).upper()
        franja = int(franja_match.group(1))

        rows = execute_query("""
                             SELECT ps.identifier
                             FROM parking_spaces ps
                                      JOIN zones z ON z.id = ps.zone_id
                             WHERE z.identifier = %s AND ps.strip_identifier = %s AND ps.status = 'free'
                             LIMIT 3;
                             """, (zona, franja))

        if not rows:
            return {"respuesta": f"No hay parqueaderos disponibles en la zona {zona}, franja {franja}."}

        parqueaderos = [r[0] for r in rows]
        return {
            "respuesta": f"En la zona {zona}, franja {franja}, los siguientes parqueaderos estan disponibles:\n- " + "\n- ".join(parqueaderos)
        }

    return {"respuesta": f"Comando reconocido: {command}"}