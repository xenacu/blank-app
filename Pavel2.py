import streamlit as st
import cv2
import tempfile
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="TDAH Dashboard", layout="wide")
st.title("📊 Analisis de TDAH con Vision por Computadora")

if "analizando" not in st.session_state:
    st.session_state.analizando = False

st.sidebar.title("🎛 Configuracion")
st.sidebar.markdown("Este analisis evaluara automaticamente las tres dimensiones del TDAH.")

nivel = st.sidebar.selectbox("📚 Nivel educativo:", ["PRIMARIA", "SECUNDARIA"])
graduacion = st.sidebar.selectbox("🎓 Grado:", ["1RO", "2DO", "3RO", "4TO", "5TO", "6TO"])

video_file = st.sidebar.file_uploader("📁 Sube un video (.mp4)", type=["mp4"])

col_start, col_stop = st.sidebar.columns(2)
with col_start:
    if st.button("🚀 Comenzar"):
        if video_file:
            st.session_state.analizando = True
with col_stop:
    if st.button("⏹️ Detener"):
        st.session_state.analizando = False

model = YOLO("yolo11s-pose.pt")
tracker_config = "botsort.yaml"
region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

tiempo_fuera = {}
eventos = []
entradas_salidas = {}
pos_anterior = {}
frame_count = 0
id_map = {}
next_id = 1

def point_in_region(x, y, region):
    return cv2.pointPolygonTest(np.array(region, dtype=np.int32), (x, y), False) >= 0
def registrar_evento(persona_id, frame, descripcion, indicador, item):
    if item not in checklist_flags[persona_id]:
        eventos.append({
            "persona_id": persona_id,
            "frame": frame,
            "evento": descripcion,
            "indicador": indicador
        })
        checklist_flags[persona_id].add(item)

if st.session_state.analizando and video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    umbral_frames = int(5 * fps)

    st.subheader("🎬 Analisis en vivo")
    col1, col2 = st.columns([3, 1])
    stframe = col1.empty()
    eventos_sidebar = col2.empty()
    col2.markdown("### 📌 Eventos Detectados")

    #Inicialización de estructuras necesarias antes del bucle principal
    tiempo_fuera = {}
    eventos = []
    entradas_salidas = {}
    pos_anterior = {}
    frame_count = 0
    id_map = {}
    next_id = 1
    checklist_flags = {}

    foco_cambios = {}
    atencion_sostenida = {}
    frente_estimulo = {}
    tarea_abandonada = {}
    organizacion_mov = {}
    evita_esfuerzo = {}
    busca_objetos = {}
    inquietud = {}
    abandono_asiento = {}
    trepa_corre = {}
    historial_pos = {}
    juego_ruidoso = {}
    actividad_excesiva = {}
    respuestas_impulsivas = {}
    espera_turno = {}
    interrupciones = {}
    habla_exceso = {}
    inicio_temprano = {}
    multi_contexto = {}
    afecta_rendimiento = {}
    descarta_otro_trastorno = {}
    detecta_estimulo = {}
    historial_proximidad = {}
    personas = {}
    personas_dentro = set()
    personas_fuera = set()
    velocidad_actual = 0
    edad_inicio = {}
    contexto_escuela = {}
    contexto_casa = {}
    eventos_por_persona = {}
    validado_sin_otro_trastorno = {}
    dist13 = {}
    dist_inquietud = {}
    esta_sentado = {}

    contexto = "aula"  # puedes modificar si estás en otro escenario
    fps = 30  # tasa de cuadros por segundo
    umbral_frames = fps * 5  # 5 segundos estándar

    frames_observados = {}

    while cap.isOpened() and st.session_state.analizando:
        success, im0 = cap.read()
        if not success:
            break

        results = model.track(im0, persist=True, tracker=tracker_config, verbose=False)[0]
        personas_dentro.clear()
        personas_fuera.clear()

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if results.names[cls_id] == "person" and box.id is not None:
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if track_id not in id_map:
                        id_map[track_id] = next_id
                        next_id += 1

                    persona_id = id_map[track_id]
                    checklist_flags.setdefault(persona_id, set())
                    frames_observados[persona_id] = frames_observados.get(persona_id, 0) + 1

                    dentro = point_in_region(cx, cy, region)
                    if dentro:
                        personas_dentro.add(persona_id)
                    else:
                        personas_fuera.add(persona_id)

                    label = f"Persona {persona_id}"
                    color = (0, 255, 0) if dentro else (0, 0, 255)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(im0, (cx, cy), 5, color, -1)

                    if frames_observados[persona_id] < fps * 3:
                        continue

                    # === DETECCIÓN COMPLETA DE ÍTEMS ===

                    # ITEM 1

                    foco_cambios.setdefault(persona_id, {"pos_anterior": (cx, cy), "cambios": 0})
                    dx = abs(cx - foco_cambios[persona_id]["pos_anterior"][0])
                    dy = abs(cy - foco_cambios[persona_id]["pos_anterior"][1])
                    dist_foco = np.sqrt(dx ** 2 + dy ** 2)
                    if 10 < dist_foco < 150:
                        foco_cambios[persona_id]["cambios"] += 1
                        print(f"[DEBUG] Persona {persona_id} - foco cambios: {foco_cambios[persona_id]['cambios']}")
                    if foco_cambios[persona_id]["cambios"] >= 6:
                        registrar_evento(persona_id, frame_count, "Cambios erraticos de foco",
                                         "Deficit de atencion - item 1", 1)
                        print(f"[REGISTRO] Item 1 activado para Persona {persona_id}")
                        foco_cambios[persona_id]["cambios"] = 0
                    foco_cambios[persona_id]["pos_anterior"] = (cx, cy)

                    # ITEM 2
                    atencion_sostenida.setdefault(persona_id, {"fuera_frames": 0})
                    if not dentro:
                        atencion_sostenida[persona_id]["fuera_frames"] += 1
                    else:
                        atencion_sostenida[persona_id]["fuera_frames"] = 0
                    if atencion_sostenida[persona_id]["fuera_frames"] >= umbral_frames * 3:
                        registrar_evento(persona_id, frame_count, "No mantiene atencion sostenida",
                                         "Deficit de atencion - item 2", 2)
                        atencion_sostenida[persona_id]["fuera_frames"] = 0

                    # ITEM 3
                    frente_estimulo.setdefault(persona_id, 0)
                    if point_in_region(cx, cy, region):
                        frente_estimulo[persona_id] += 1
                    if frente_estimulo[persona_id] == 0 and frame_count > umbral_frames * 2:
                        registrar_evento(persona_id, frame_count, "No responde a estimulo visual",
                                         "Deficit de atencion - item 3", 3)

                    # ITEM 4
                    tarea_abandonada.setdefault(persona_id, 0)
                    if not dentro:
                        tarea_abandonada[persona_id] += 1
                    if tarea_abandonada[persona_id] >= 15:
                        registrar_evento(persona_id, frame_count, "Abandona tarea frecuentemente",
                                         "Deficit de atencion - item 4", 4)
                        tarea_abandonada[persona_id] = 0

                    # ITEM 5
                    organizacion_mov.setdefault(persona_id, {"cambios": 0, "pos": (cx, cy)})
                    dx5 = abs(cx - organizacion_mov[persona_id]["pos"][0])
                    dy5 = abs(cy - organizacion_mov[persona_id]["pos"][1])
                    dist_mov = np.sqrt(dx5 ** 2 + dy5 ** 2)
                    if dist_mov > 100:
                        organizacion_mov[persona_id]["cambios"] += 1
                    if organizacion_mov[persona_id]["cambios"] >= 10:
                        registrar_evento(persona_id, frame_count, "Movimiento desorganizado constante",
                                         "Deficit de atencion - item 5", 5)
                        organizacion_mov[persona_id]["cambios"] = 0
                    organizacion_mov[persona_id]["pos"] = (cx, cy)

                    # ITEM 6 (Evita esfuerzo sostenido - corregido)
                    evita_esfuerzo.setdefault(persona_id, 0)
                    if not dentro:
                        evita_esfuerzo[persona_id] += 1
                    if evita_esfuerzo[persona_id] >= 60:
                        registrar_evento(persona_id, frame_count, "Evita tareas con esfuerzo sostenido",
                                         "Deficit de atencion - item 6", 6)
                        evita_esfuerzo[persona_id] = 0

                    # ITEM 7 (Busca objetos o se distrae)
                    busca_objetos.setdefault(persona_id, 0)
                    if not dentro:
                        busca_objetos[persona_id] += 1
                    if busca_objetos[persona_id] >= 25:  # antes 10
                        registrar_evento(persona_id, frame_count, "Busca objetos o sale del area",
                                         "Deficit de atencion - item 7", 7)
                        busca_objetos[persona_id] = 0

                    # ITEM 8 (Fuera de la región > 10s)
                    tiempo_fuera.setdefault(persona_id, 0)
                    if persona_id in personas_fuera:
                        tiempo_fuera[persona_id] += 1
                        if tiempo_fuera[persona_id] >= umbral_frames * 2:
                            registrar_evento(persona_id, frame_count, "Fuera de region > 10s",
                                             "Deficit de atencion - item 8", 8)
                            tiempo_fuera[persona_id] = 0
                    else:
                        tiempo_fuera[persona_id] = 0

                    # ITEM 9 (Inactividad prolongada sin eventos recientes)
                    if 9 not in checklist_flags[persona_id]:
                        eventos_persona = [ev for ev in eventos if ev["persona_id"] == persona_id]
                        if eventos_persona and dentro:
                            ult_evento = eventos_persona[-1]["frame"]
                            if frame_count - ult_evento >= umbral_frames * 4:  # antes 3
                                registrar_evento(persona_id, frame_count, "Inactividad prolongada (posible olvido)",
                                                 "Deficit de atencion - item 9", 9)
                                checklist_flags[persona_id].add(9)

                    # ITEM 10 (Inquietud motora)
                    inquietud.setdefault(persona_id, {"movimientos": 0, "pos": (cx, cy)})  # Inicializa si no existe

                    dx10 = abs(cx - inquietud[persona_id]["pos"][0])
                    dy10 = abs(cy - inquietud[persona_id]["pos"][1])
                    dist_inquietud = np.sqrt(dx10 ** 2 + dy10 ** 2)

                    if 10 < dist_inquietud < 50:
                        inquietud[persona_id]["movimientos"] += 1

                    if inquietud[persona_id]["movimientos"] >= 30:
                        registrar_evento(persona_id, frame_count, "Inquietud motora en asiento",
                                         "Hiperactividad - item 10", 10)
                        inquietud[persona_id]["movimientos"] = 0

                    # Actualiza posición
                    inquietud[persona_id]["pos"] = (cx, cy)

                    # ITEM 11 (Abandona el asiento sin motivo)
                    abandono_asiento.setdefault(persona_id, 0)
                    esta_sentado.setdefault(persona_id, True)  # Se asume que inicialmente está sentado

                    # Si estaba sentado y ahora no está dentro del área, contar como intento de abandono
                    if esta_sentado[persona_id] and not dentro:
                        abandono_asiento[persona_id] += 1
                    else:
                        abandono_asiento[persona_id] = 0  # Reinicia si volvió a sentarse o nunca estuvo sentado

                    # Si excede el umbral, registrar el evento
                    if abandono_asiento[persona_id] >= 8:
                        registrar_evento(persona_id, frame_count, "Abandona el asiento sin motivo",
                                         "Hiperactividad - item 11", 11)
                        abandono_asiento[persona_id] = 0

                    # ITEM 12 (Corretea o trepa en exceso)
                    trepa_corre.setdefault(persona_id, 0)
                    if velocidad_actual > 2.8:
                        trepa_corre[persona_id] += 1
                    if trepa_corre[persona_id] >= 12:
                        registrar_evento(persona_id, frame_count, "Corretea o trepa en exceso",
                                         "Hiperactividad - item 12", 12)
                        trepa_corre[persona_id] = 0

                    # ITEM 13 (Juego ruidoso o sin control)
                    juego_ruidoso.setdefault(persona_id, {"movs": 0, "zona": (cx, cy)})

                    dx13 = abs(cx - juego_ruidoso[persona_id]["zona"][0])
                    dy13 = abs(cy - juego_ruidoso[persona_id]["zona"][1])
                    dist13 = np.sqrt(dx13 ** 2 + dy13 ** 2)

                    if dist13 > 90:  # antes 80
                        juego_ruidoso[persona_id]["movs"] += 1
                    if juego_ruidoso[persona_id]["movs"] >= 10:  # antes 8
                        registrar_evento(persona_id, frame_count, "Juego ruidoso o sin control",
                                         "Hiperactividad - item 13", 13)
                        juego_ruidoso[persona_id]["movs"] = 0

                    # Actualizar posición
                    juego_ruidoso[persona_id]["zona"] = (cx, cy)

                    # ITEM 14 (Actividad excesiva persistente)
                    actividad_excesiva.setdefault(persona_id, 0)

                    if velocidad_actual > 2.8:
                        actividad_excesiva[persona_id] += 1
                    else:
                        actividad_excesiva[persona_id] = max(0, actividad_excesiva[persona_id] - 1)

                    if actividad_excesiva[persona_id] >= 30:
                        registrar_evento(persona_id, frame_count, "Actividad excesiva persistente",
                                         "Hiperactividad - item 14", 14)
                        actividad_excesiva[persona_id] = 0

                    # ITEM 15
                    respuestas_impulsivas.setdefault(persona_id, 0)
                    if detecta_estimulo and velocidad_actual > 2.5:
                        respuestas_impulsivas[persona_id] += 1
                    if respuestas_impulsivas[persona_id] >= 3:
                        registrar_evento(persona_id, frame_count, "Responde sin esperar pregunta completa",
                                         "Impulsividad - item 15", 15)
                        respuestas_impulsivas[persona_id] = 0

                    # ITEM 16 - lógica mejorada basada en proximidad a otros (impaciencia / interrupción)
                    espera_turno.setdefault(persona_id, 0)
                    interaccion_detectada = False

                    for otra_id, otra_pos in personas.items():
                        if otra_id == persona_id:
                            continue
                        dist = np.sqrt((cx - otra_pos["cx"]) ** 2 + (cy - otra_pos["cy"]) ** 2)
                        if dist < 60:  # Se aproxima demasiado a otra persona
                            espera_turno[persona_id] += 1
                            interaccion_detectada = True
                            break

                    if not interaccion_detectada:
                        espera_turno[persona_id] = max(0, espera_turno[persona_id] - 1)

                    if espera_turno[persona_id] >= 5:
                        registrar_evento(persona_id, frame_count, "Dificultad para esperar su turno",
                                         "Impulsividad - item 16", 16)
                        espera_turno[persona_id] = 0

                    # ITEM 17
                    interrupciones.setdefault(persona_id, 0)
                    historial_proximidad.setdefault(persona_id, {"anterior": (cx, cy)})
                    se_aproxima = False
                    for otra_id, otra_pos in personas.items():
                        if otra_id == persona_id:
                            continue
                        dist_actual = np.sqrt((cx - otra_pos["cx"]) ** 2 + (cy - otra_pos["cy"]) ** 2)
                        prev = historial_proximidad[persona_id]["anterior"]
                        dist_prev = np.sqrt((prev[0] - otra_pos["cx"]) ** 2 + (prev[1] - otra_pos["cy"]) ** 2)
                        if dist_prev > dist_actual and dist_actual < 100:
                            se_aproxima = True
                            break
                    historial_proximidad[persona_id]["anterior"] = (cx, cy)
                    if se_aproxima:
                        interrupciones[persona_id] += 1
                    if interrupciones[persona_id] >= 3:
                        registrar_evento(persona_id, frame_count, "Interrupciones frecuentes a otros",
                                         "Impulsividad - item 17", 17)
                        interrupciones[persona_id] = 0

                    # ITEM 18
                    habla_exceso.setdefault(persona_id, 0)
                    if velocidad_actual < 0.3 and dentro:
                        habla_exceso[persona_id] += 1
                    else:
                        habla_exceso[persona_id] = max(0, habla_exceso[persona_id] - 1)
                    if habla_exceso[persona_id] >= 30:
                        registrar_evento(persona_id, frame_count, "Habla en exceso sin moderarse",
                                         "Impulsividad - item 18", 18)
                        habla_exceso[persona_id] = 0

                    # ITEM 19 al 22 se basan en datos clínicos o registros externos
                    #if edad_inicio.get(persona_id, 99) < 7:
                     #   registrar_evento(persona_id, frame_count, "Inicio antes de los 7 años",
                                        # "Impulsividad - item 19", 19)
                    #if contexto_escuela.get(persona_id) and contexto_casa.get(persona_id):
                        #registrar_evento(persona_id, frame_count, "Conductas en múltiples contextos",
                                         #"Impulsividad - item 20", 20)
                    #if len(eventos_por_persona.get(persona_id, [])) >= 10:
                    #    registrar_evento(persona_id, frame_count, "Síntomas afectan rendimiento",
                    #                     #"Impulsividad - item 21", 21)
                    #if validado_sin_otro_trastorno.get(persona_id):
                    #    registrar_evento(persona_id, frame_count, "No cumple otros diagnósticos",
                                         #"Impulsividad - item 22", 22)

                    total_eventos = len(checklist_flags.get(persona_id, set()))
                    if total_eventos >= 2:
                        cv2.putText(im0, "\u26A0 Atencion", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                                    2)

        stframe.image(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        frame_count += 1

        eventos_sidebar.markdown("**Eventos recientes:**")
        for e in eventos[-5:]:
            info = f"Persona {e['persona_id']} — {e['evento']}"
            if "indicador" in e:
                info += f" ({e['indicador']})"
            eventos_sidebar.info(info)

    cap.release()
    st.success(f"✅ Analisis completo. Eventos detectados: {len(eventos)}")
    st.download_button("⬇️ Descargar registro de eventos", json.dumps(eventos, indent=2), file_name="eventos.json", mime="application/json")

st.markdown("---")

st.header("🧠 Diagnostico Preliminar por Individuo")

data_export = []
personas = {}
indicadores = {}

for e in eventos:
    pid = e["persona_id"]
    if pid not in personas:
        personas[pid] = {"Deficit de atencion": 0, "Hiperactividad": 0, "Impulsividad": 0}
        indicadores[pid] = {"Deficit de atencion": [], "Hiperactividad": [], "Impulsividad": []}

    evento = e["evento"]

    if evento == "Cambios erraticos de foco":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 1")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 1,
                            "PREGUNTA": "Tiene dificultad para mantener el foco en tareas o actividades?",
                            "RESPUESTA": "SI"})

    elif evento == "No mantiene atencion sostenida":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 2")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 2,
                            "PREGUNTA": "No presta atencion durante actividades o juegos?", "RESPUESTA": "SI"})

    elif evento == "No responde a estimulo visual":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 3")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 3,
                            "PREGUNTA": "No parece escuchar cuando se le habla directamente?", "RESPUESTA": "SI"})

    elif evento == "Abandona tarea frecuentemente":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 4")
        data_export.append(
            {"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 4, "PREGUNTA": "Abandona tareas sin terminarlas?",
             "RESPUESTA": "SI"})

    elif evento == "Movimiento desorganizado constante":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 5")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 5,
                            "PREGUNTA": "Tiene dificultades para organizar tareas o actividades?", "RESPUESTA": "SI"})

    elif evento == "Evita tareas con esfuerzo sostenido":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 6")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 6,
                            "PREGUNTA": "Evita o rechaza tareas que requieren esfuerzo mental sostenido?",
                            "RESPUESTA": "SI"})

    elif evento == "Busca objetos o sale del area":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 7")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 7,
                            "PREGUNTA": "Pierde objetos necesarios para tareas?", "RESPUESTA": "SI"})

    elif evento == "Fuera de region > 5s":
        personas[pid]["Deficit de atencion"] += 1
        indicadores[pid]["Deficit de atencion"].append("item 8")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 8,
                            "PREGUNTA": "Se distrae facilmente por estimulos externos?", "RESPUESTA": "SI"})

    # --- Ítems 10 al 14: Hiperactividad ---

    elif evento == "Inquietud motora en asiento":
        personas[pid]["Hiperactividad"] += 1
        indicadores[pid]["Hiperactividad"].append("item 10")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 10,
                            "PREGUNTA": "¿Se muestra inquieto, moviendo manos o pies, o moviéndose en el asiento?",
                            "RESPUESTA": "SI"})

    elif evento == "Abandona el asiento sin motivo":
        personas[pid]["Hiperactividad"] += 1
        indicadores[pid]["Hiperactividad"].append("item 11")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 11,
                            "PREGUNTA": "¿Abandona el asiento en situaciones donde se espera que permanezca sentado?",
                            "RESPUESTA": "SI"})

    elif evento == "Corretea o trepa en exceso":
        personas[pid]["Hiperactividad"] += 1
        indicadores[pid]["Hiperactividad"].append("item 12")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 12,
                            "PREGUNTA": "¿Corretea o trepa en exceso en situaciones inapropiadas?",
                            "RESPUESTA": "SI"})

    elif evento == "Juego ruidoso o sin control":
        personas[pid]["Hiperactividad"] += 1
        indicadores[pid]["Hiperactividad"].append("item 13")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 13,
                            "PREGUNTA": "¿Es ruidoso de manera inapropiada durante el juego o tiene dificultades para entretenerse tranquilamente?",
                            "RESPUESTA": "SI"})

    elif evento == "Actividad excesiva persistente":
        personas[pid]["Hiperactividad"] += 1
        indicadores[pid]["Hiperactividad"].append("item 14")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 14,
                            "PREGUNTA": "¿Muestra un patrón persistente de actividad excesiva que no se ajusta a las demandas del entorno social?",
                            "RESPUESTA": "SI"})

    # --- Ítems 15 al 22: Impulsividad ---

    elif evento == "Responde sin esperar pregunta completa":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 15")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 15,
                            "PREGUNTA": "¿Responde o exclama antes de que las preguntas sean completadas?",
                            "RESPUESTA": "SI"})

    elif evento == "Dificultad para esperar su turno":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 16")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 16,
                            "PREGUNTA": "¿Tiene dificultad para esperar su turno en colas u otras situaciones grupales?",
                            "RESPUESTA": "SI"})

    elif evento == "Interrupciones frecuentes a otros":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 17")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 17,
                            "PREGUNTA": "¿Interrumpe frecuentemente los asuntos de otros?",
                            "RESPUESTA": "SI"})

    elif evento == "Habla en exceso sin moderarse":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 18")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 18,
                            "PREGUNTA": "¿Habla en exceso sin moderarse en situaciones sociales?",
                            "RESPUESTA": "SI"})

    elif evento == "Inicio antes de los 7 años":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 19")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 19,
                            "PREGUNTA": "¿El inicio de estos comportamientos fue antes de los siete años?",
                            "RESPUESTA": "SI"})

    elif evento == "Conductas en múltiples contextos":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 20")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 20,
                            "PREGUNTA": "¿Estos comportamientos se observan en más de un contexto?",
                            "RESPUESTA": "SI"})

    elif evento == "Síntomas afectan rendimiento":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 21")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 21,
                            "PREGUNTA": "¿Estos síntomas afectan el rendimiento social, académico o laboral?",
                            "RESPUESTA": "SI"})

    elif evento == "No cumple otros diagnósticos":
        personas[pid]["Impulsividad"] += 1
        indicadores[pid]["Impulsividad"].append("item 22")
        data_export.append({"ID": pid, "NIVEL": nivel, "GRADO": graduacion, "ITEM": 22,
                            "PREGUNTA": "¿No cumple con criterios para otro trastorno como TGD, manía, depresión o ansiedad?",
                            "RESPUESTA": "SI"})


for pid, dim in personas.items():
    st.subheader(f"🧍 Persona {pid}")
    da, ha, im = dim["Deficit de atencion"], dim["Hiperactividad"], dim["Impulsividad"]
    total = da + ha + im

    st.markdown(f"""
    - **Déficit de Atención**: {da} positivo(s) → {"✅ Cumple" if da >= 5 else "❌ No cumple"} | Detectado(s): {', '.join(indicadores[pid]['Deficit de atencion']) or 'Ninguno'}
    - **Hiperactividad**: {ha} positivo(s) → {"✅ Cumple" if ha >= 2 else "❌ No cumple"} | Detectado(s): {', '.join(indicadores[pid]['Hiperactividad']) or 'Ninguno'}
    - **Impulsividad**: {im} positivo(s) → {"✅ Cumple" if im >= 5 else "❌ No cumple"} | Detectado(s): {', '.join(indicadores[pid]['Impulsividad']) or 'Ninguno'}
    - **Total ítems detectados**: {total}
    """)

    if total >= 4:
        if da >= 6 and (ha >= 3 or im >= 4):
            st.warning("📕 Posible TDAH tipo combinado")
        elif da >= 6:
            st.info("📘 Posible TDAH tipo inatento")
        elif ha >= 3 or im >= 4:
            st.info("📗 Posible TDAH tipo hiperactivo/impulsivo")
        else:
            st.success("🔍 Posible TDAH - no se ajusta a subtipo clásico")
    else:
        st.success("🟢 No se detectan criterios suficientes")


if data_export:
    df_export = pd.DataFrame(data_export)
    csv = df_export.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    df_export.to_excel(excel_buffer, index=False, sheet_name="Diagnostico")
    st.download_button("⬇️ Descargar CSV", csv, "diagnostico_tdah.csv", "text/csv")
    st.download_button("⬇️ Descargar Excel", excel_buffer.getvalue(), "diagnostico_tdah.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Exportar a PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Reporte de TDAH por Vision por Computadora", ln=1, align="C")
    pdf.ln(10)
    # Agrupar por ítem
    df_export = pd.DataFrame(data_export)
    items_agrupados = df_export.groupby("ITEM")

    for item, grupo in items_agrupados:
        pregunta = grupo["PREGUNTA"].iloc[0]
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 10, txt=f"Ítem {item}: {pregunta}")
        pdf.set_font("Arial", size=10)
        for _, row in grupo.iterrows():
            linea = f"  - Persona ID: {row['ID']} | Nivel: {row['NIVEL']} | Grado: {row['GRADO']} | Respuesta: {row['RESPUESTA']}"
            pdf.multi_cell(0, 10, txt=linea)
        pdf.ln(5)  # Espacio entre ítems

    pdf_output = pdf.output(dest='S').encode('latin-1')
    st.download_button("⬇️ Descargar PDF", pdf_output, file_name="diagnostico_tdah.pdf", mime="application/pdf")