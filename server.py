"""
Servidor Flask para procesamiento de consultas NLP con Anthropic API
Analista de datos petroleros - Cuenca del Golfo San Jorge
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import csv
import io
import json
import os
import re
import requests as req
import yfinance as yf
import pandas as pd
import hashlib

import time

# ── Demo ──────────────────────────────────────────────────────────
DEMO_PARQUET = os.path.join(os.path.dirname(
    __file__), 'data', 'produccion_pozos.parquet')
_demo_empresas_cache = None  # Solo la lista de empresas, liviana

# ── Parquet cache ─────────────────────────────────────────────────────
# NOTA: Railway tiene filesystem efímero — el cache se pierde al reiniciar.
# Para persistencia real, usar un volumen Railway o almacenamiento externo.
PARQUET_CACHE_DIR = './cache'
os.makedirs(PARQUET_CACHE_DIR, exist_ok=True)

CAT_COLS = [
    "sigla", "formprod", "formacion",
    "areapermisoconcesion", "areayacimiento", "cuenca",
    "provincia", "tipo_de_recurso", "proyecto",
    "clasificacion", "subclasificacion", "sub_tipo_recurso",
    "tipoextraccion", "tipoestado", "tipopozo"
]


def optimizar_df(df):
    """Reduce memoria aplicando categoricals y downcast numérico."""
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def get_parquet_path(file_hash):
    return os.path.join(PARQUET_CACHE_DIR, f"{file_hash}.parquet")


def leer_csv_y_cachear(file_bytes, file_hash):
    """Lee el CSV desde bytes, optimiza y guarda Parquet. Devuelve DataFrame."""
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        dtype=str,
        low_memory=False,
        encoding='utf-8-sig'
    )
    df = optimizar_df(df)
    parquet_path = get_parquet_path(file_hash)
    df.to_parquet(parquet_path, index=False)
    return df


_cache = {}
CACHE_TTL = 300  # 5 minutos en segundos


def get_cached(key):
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None


def set_cached(key, data):
    _cache[key] = (data, time.time())


app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 400 * 1024 * 1024  # 400 MB

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def get_ai_client():
    """Obtiene el cliente OpenAI usando la API Key enviada desde el frontend."""
    # Busca la clave en los headers, si no está, usa la variable de entorno
    req_key = request.headers.get('X-API-Key')
    key_to_use = req_key if req_key else API_KEY

    if not key_to_use:
        raise ValueError(
            "Clave API no configurada. Por favor, ingrésala en la configuración.")

    return OpenAI(api_key=key_to_use, base_url="https://api.deepseek.com")


EXPECTED_SCHEMAS = {
    "historico": [
        "pozo",
        "fecha",
        "area",
        "estado",
        "produccion",
        "prod_agua",
        "prod_gas",
        "corte_agua",
        "tef"
    ],
    "catalogo": [
        "pozo",
        "area",
        "nombre_area",
        "formacion",
        "tipo",
        "año_perforacion"
    ]
}

SYSTEM_PROMPT = """Eres un analista experto en datos petroleros de Argentina.

Tu tarea es analizar datos de producción de pozos y responder consultas en lenguaje natural.

DATOS DISPONIBLES:
- Columnas: pozo, fecha, area, cuenca, estado, produccion (m³/mes, petróleo neto), prod_agua (m³/mes), prod_gas (miles de m³/mes), corte_agua (%), tef (días efectivos de flujo)
- Estados posibles: Activo, Parado, Mantenimiento

INSTRUCCIONES:
1. Responde en español de forma concisa y profesional
2. Incluye datos numéricos relevantes cuando corresponda
3. Si la consulta es ambigua, pide clarificación
4. Usa formato markdown: **negrita**, tablas con |, ## títulos
5. No inventes datos - usa solo la información proporcionada

FORMATO DE RESPUESTA:
- Sé directo y ve al punto
- Incluye cifras clave (producción petróleo, gas, porcentajes, presiones)
- Si comparas, usá tablas markdown
- Máximo 3-4 oraciones para consultas simples"""


def build_context(summary, catalogo_data):
    """Construye contexto de datos para el prompt usando el resumen"""
    context = "DATOS CARGADOS:\n\n"

    if summary:
        context += "RESUMEN GENERAL:\n"
        context += f"- Total registros: {summary.get('totalRecords', 0)}\n"
        context += f"- Pozos únicos: {summary.get('uniqueWells', 0)}\n"

        latest_records = summary.get('latestRecords', [])
        if latest_records:
            context += "\nÚLTIMOS REGISTROS POR POZO:\n"
            for record in latest_records:
                pozo = record.get('pozo', 'N/A')
                area = record.get('area', 'N/A')
                estado = record.get('estado', 'N/A')
                prod = record.get('produccion', 0) or 0
                gas = record.get('prod_gas', 0) or 0
                agua = record.get('corte_agua', 0) or 0
                tef = record.get('tef', 0) or 0
                context += f"- {pozo}: {area} | Estado: {estado} | "
                context += f"Producción petróleo: {prod:.2f} m³/día | "
                context += f"Producción gas: {gas:.2f} miles m³ | "
                context += f"Corte agua: {agua:.1f}% | "
                context += f"TEF: {tef:.0f} días\n"

        area_production = summary.get('areaProduction', {})
        if area_production:
            context += "\nPRODUCCIÓN POR ÁREA:\n"
            for area, prod in area_production.items():
                if area and area != 'undefined':
                    context += f"- {area}: {prod:.2f} m³/día total\n"

    if catalogo_data:
        context += "\nCATÁLOGO DE POZOS:\n"
        for pozo in catalogo_data:
            context += f"- {pozo.get('pozo', 'N/A')}: "
            context += f"Área: {pozo.get('area', 'N/A')} | "
            context += f"Formación: {pozo.get('formacion', 'N/A')} | "
            context += f"Tipo: {pozo.get('tipo', 'N/A')} | "
            context += f"Perforado: {pozo.get('año_perforacion', 'N/A')}\n"

    return context


def parse_mapping_json(raw_text):
    """Extrae y parsea JSON de mapeo desde la respuesta del modelo."""
    if not raw_text:
        raise ValueError("Respuesta vacía del modelo")

    stripped = raw_text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("No se encontró JSON válido en la respuesta")
        return json.loads(match.group(0))


def normalize_csv_text(csv_text, mapping, expected_schema):
    """Renombra columnas de un CSV completo usando mapping."""
    reader = csv.DictReader(io.StringIO(csv_text))
    original_fields = reader.fieldnames or []
    rows = list(reader)

    renamed_rows = []
    for row in rows:
        renamed = {}
        for original_col, value in row.items():
            target_col = mapping.get(original_col)
            if target_col in expected_schema:
                renamed[target_col] = value
        renamed_rows.append(renamed)

    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=expected_schema, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(renamed_rows)

    missing_required = [
        col for col in expected_schema if col not in mapping.values()]
    return output.getvalue(), original_fields, missing_required


@app.route('/query', methods=['POST'])
def process_query():
    """Endpoint para procesar consultas NLP"""
    try:
        client = get_ai_client()
        data = request.json
        query = data.get('query', '')
        summary = data.get('summary', {})
        catalogo_data = data.get('catalogo', [])

        if not query:
            return jsonify({'error': 'Consulta vacía'}), 400

        context = build_context(summary, catalogo_data)

        user_prompt = f"""{context}

CONSULTA DEL USUARIO: "{query}"

Responde la consulta basándote en los datos proporcionados arriba."""

        message = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )

        return jsonify({
            'success': True,
            'response': message.choices[0].message.content,
            'model': 'deepseek-chat'
        })

    except Exception as e:
        app.logger.error(f"Error procesando consulta: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/normalize', methods=['POST'])
def normalize():
    """Normaliza columnas CSV al esquema esperado usando Claude."""
    try:
        client = get_ai_client()
        data = request.json or {}
        csv_text = data.get('csv_text', '')
        tipo = (data.get('tipo', '') or '').strip().lower()

        if not csv_text:
            return jsonify({'success': False, 'error': 'csv_text es requerido'}), 400

        if tipo not in EXPECTED_SCHEMAS:
            return jsonify({'success': False, 'error': 'tipo debe ser historico o catalogo'}), 400

        expected_schema = EXPECTED_SCHEMAS[tipo]
        reader = csv.DictReader(io.StringIO(csv_text))
        columns = reader.fieldnames or []
        sample_rows = []
        for idx, row in enumerate(reader):
            if idx >= 3:
                break
            sample_rows.append(row)

        if not columns:
            return jsonify({'success': False, 'error': 'No se encontraron columnas en el CSV'}), 400

        user_prompt = (
            f"Estas son las columnas del CSV recibido: {columns}.\n"
            f"El esquema esperado es: {expected_schema}.\n"
            f"Muestra de las primeras 3 filas: {sample_rows}.\n"
            "Devolvé SOLO un JSON con el mapeo de columnas, formato: "
            "{\"columna_original\": \"columna_esperada\"}. "
            "Si una columna no tiene equivalente, mapeala a null."
        )

        message = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=500,
            messages=[
                {"role": "system", "content": "Sos un asistente de normalización de columnas CSV. Respondé solo JSON válido."},
                {"role": "user", "content": user_prompt}
            ],
        )

        raw_mapping = message.choices[0].message.content if message.choices else ""
        mapping = parse_mapping_json(raw_mapping)

        if not isinstance(mapping, dict):
            return jsonify({'success': False, 'error': 'El mapeo devuelto no es un objeto JSON'}), 500

        normalized_csv, original_fields, missing_required = normalize_csv_text(
            csv_text,
            mapping,
            expected_schema
        )

        if missing_required:
            return jsonify({
                'success': False,
                'error': f'No se pudieron mapear columnas requeridas: {", ".join(missing_required)}',
                'missing_required': missing_required,
                'mapping': mapping,
                'original_columns': original_fields
            }), 422

        return jsonify({
            'success': True,
            'normalized_csv': normalized_csv,
            'mapping': mapping,
            'original_columns': original_fields,
            'schema': expected_schema
        })

    except Exception as e:
        app.logger.error(f"Error normalizando CSV: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


def calcular_corte_agua(prod_pet, prod_agua):
    try:
        pet = float(prod_pet or 0)
        agua = float(prod_agua or 0)
        total = pet + agua
        if total == 0:
            return 0.0
        return round(agua / total * 100, 1)
    except:
        return 0.0


def normalizar_estado(tipoestado):
    estado = str(tipoestado).strip()
    activos = {"Extracción Efectiva", "En Inyección Efectiva"}
    mant = {"En Estudio", "En Reserva para Recuperación Secundaria/Asistida"}
    if estado in activos:
        return "Activo"
    elif estado in mant:
        return "Mantenimiento"
    else:
        return "Parado"


def procesar_df(df):
    """
    Versión vectorizada de procesar_df.
    ~50x más rápida que iterrows() para DataFrames grandes.
    """
    d = df.copy()

    # ── Columnas de texto: fillna + strip ────────────────────────────
    def col_str(nombre):
        if nombre not in d.columns:
            return pd.Series('', index=d.index)
        return d[nombre].fillna('').astype(str).str.strip()

    idpozo = col_str('idpozo')
    sigla = col_str('sigla')
    # Si sigla está vacía, cae a idpozo
    sigla = sigla.where(sigla != '', idpozo)

    anio = col_str('anio')
    mes = col_str('mes').str.zfill(2)

    # ── Fecha vectorizada ─────────────────────────────────────────────
    mask_fecha = (anio != '') & (mes != '') & (anio != 'nan') & (mes != 'nan')
    fecha = pd.Series('', index=d.index)
    fecha[mask_fecha] = anio[mask_fecha] + '-' + mes[mask_fecha]

    # ── Columnas numéricas ────────────────────────────────────────────
    def col_num(nombre):
        if nombre not in d.columns:
            return pd.Series(0.0, index=d.index)
        return pd.to_numeric(d[nombre], errors='coerce').fillna(0.0)

    prod_pet = col_num('prod_pet')
    prod_agua = col_num('prod_agua')
    prod_gas = col_num('prod_gas')
    tef = col_num('tef')

    # ── Corte de agua vectorizado ─────────────────────────────────────
    total_liq = prod_pet + prod_agua
    corte = (prod_agua / total_liq.where(total_liq > 0)).fillna(0.0) * 100
    corte = corte.round(1)

    # ── Histórico ─────────────────────────────────────────────────────
    df_hist = pd.DataFrame({
        'pozo':       sigla,
        'fecha':      fecha,
        'area':       col_str('areayacimiento'),
        'cuenca':     col_str('cuenca'),
        'estado':     col_str('tipoestado').apply(normalizar_estado),
        'produccion': prod_pet.round(2),
        'prod_agua':  prod_agua.round(2),
        'prod_gas':   prod_gas.round(2),
        'corte_agua': corte,
        'tef':        tef.round(1),
    })
    historico = df_hist.to_dict(orient='records')

    # ── Catálogo (un registro por pozo único) ─────────────────────────
    df_cat = pd.DataFrame({
        'idpozo':          idpozo,
        'pozo':            sigla,
        'area':            col_str('areayacimiento'),
        'nombre_area':     col_str('areapermisoconcesion'),
        'formacion':       col_str('formacion'),
        'tipo':            col_str('tipoextraccion'),
        'año_perforacion': col_str('profundidad'),
        'empresa':         col_str('empresa'),
        'cuenca':          col_str('cuenca'),
        'provincia':       col_str('provincia'),
        'tipo_recurso':    col_str('tipo_de_recurso'),
    })
    df_cat = df_cat.drop_duplicates(subset=['idpozo'])
    catalogo = df_cat.drop(columns=['idpozo']).to_dict(orient='records')

    return historico, catalogo


@app.route('/api/procesar-csv', methods=['POST'])
def procesar_csv():
    """
    Recibe el CSV grande como archivo multipart.
    Convierte a Parquet optimizado (cache) y devuelve lista de empresas.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ningún archivo'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo vacío'}), 400

        file_bytes = file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        parquet_path = get_parquet_path(file_hash)

        if os.path.exists(parquet_path):
            # Cache hit — leer Parquet (mucho más rápido que CSV)
            df = pd.read_parquet(parquet_path, columns=[
                                 'idempresa', 'empresa'])
            app.logger.info(f"Cache hit: {parquet_path}")
        else:
            # Cache miss — parsear CSV completo, optimizar y guardar Parquet
            try:
                df_full = leer_csv_y_cachear(file_bytes, file_hash)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'El CSV no contiene las columnas idempresa o empresa'
                }), 422
            df = df_full[['idempresa', 'empresa']
                         ] if 'idempresa' in df_full.columns else df_full

        df['idempresa'] = df['idempresa'].astype(
            str).str.strip().replace('nan', '')
        df['empresa'] = df['empresa'].astype(
            str).str.strip().replace('nan', '')

        empresas = (
            df[df['idempresa'].notna() & (df['idempresa'] != '')
               & (df['idempresa'] != 'nan')]
            .drop_duplicates(subset='idempresa')
            .sort_values('empresa')
            .to_dict(orient='records')
        )

        return jsonify({'success': True, 'empresas': empresas, 'file_hash': file_hash})

    except Exception as e:
        app.logger.error(f"Error procesando CSV: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/filtrar', methods=['POST'])
def filtrar():
    """
    Recibe el archivo (CSV o hash) + idempresa.
    Usa Parquet cacheado si existe; si no, procesa el CSV y lo cachea.
    """
    try:
        idempresa = request.form.get('idempresa', '').strip()
        if not idempresa:
            return jsonify({'success': False, 'error': 'idempresa requerido'}), 400

        # Intentar usar hash del cache (evita resubir el archivo)
        file_hash = request.form.get('file_hash', '').strip()
        parquet_path = get_parquet_path(file_hash) if file_hash else None

        if parquet_path and os.path.exists(parquet_path):
            # Cache hit — leer Parquet directo
            app.logger.info(f"filtrar: cache hit {parquet_path}")
            df = pd.read_parquet(parquet_path)
        else:
            # Cache miss — necesita el archivo
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No se recibió archivo ni hash válido'}), 400

            file = request.files['file']
            file_bytes = file.read()
            computed_hash = hashlib.md5(file_bytes).hexdigest()
            computed_path = get_parquet_path(computed_hash)

            if os.path.exists(computed_path):
                df = pd.read_parquet(computed_path)
            else:
                df = leer_csv_y_cachear(file_bytes, computed_hash)

        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(str)

        df['idempresa'] = df['idempresa'].str.strip().replace('nan', '')
        df_empresa = df[df['idempresa'] == idempresa]

        if df_empresa.empty:
            return jsonify({
                'success': False,
                'error': f'No se encontraron datos para idempresa={idempresa}'
            }), 404

        historico, catalogo = procesar_df(df_empresa)

        return jsonify({
            'success':   True,
            'historico': historico,
            'catalogo':  catalogo,
            'empresa':   idempresa,
            'total':     len(historico),
            'pozos':     len(catalogo),
        })

    except Exception as e:
        app.logger.error(f"Error filtrando CSV: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/api/wfs/yacimientos')
def wfs_yacimientos():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    params = {
        'SERVICE': 'WFS',
        'VERSION': '1.1.0',
        'REQUEST': 'GetFeature',
        'TYPENAME': 'planosbase_yacimientos',
        'OUTPUTFORMAT': 'application/json; subtype=geojson',
        'MAXFEATURES': '1000'
    }
    try:
        r = req.get(
            'https://sig.energia.gob.ar/mapserv?map=/etc/mapserver/mapase.map',
            params=params,
            timeout=30,
            verify=False
        )
        content = r.content.decode('utf-8', errors='replace')
        app.logger.warning(
            f'WFS status {r.status_code} - inicio: {content[:300]}')
        if content.strip().startswith('{'):
            return app.response_class(
                response=r.content,
                status=200,
                mimetype='application/json'
            )
        return jsonify({'error': 'WFS no devolvió GeoJSON', 'respuesta': content[:500]}), 502
    except Exception as e:
        app.logger.error(f'WFS error: {e}')
        return jsonify({'error': str(e)}), 500


def is_rate_limit_error(e):
    return 'Too Many Requests' in str(e) or 'Rate limited' in str(e) or '429' in str(e)


@app.route('/api/precio-brent', methods=['GET'])
def precio_brent():
    try:
        period = request.args.get('period', '1mo')
        cache_key = f'brent_{period}'

        cached = get_cached(cache_key)
        if cached:
            return jsonify(cached)

        brent = yf.Ticker("BZ=F")
        hist = brent.history(period=period)

        if not hist.empty:
            ultimo_precio = round(float(hist['Close'].iloc[-1]), 2)
            open_price = round(float(hist['Open'].iloc[-1]), 2)
            high_price = round(float(hist['High'].max()), 2)
            low_price = round(float(hist['Low'].min()), 2)
            primer_precio = float(hist['Close'].iloc[0])
            change_pct = round(((ultimo_precio - primer_precio) /
                               primer_precio) * 100, 2) if primer_precio > 0 else 0.0
            history_data = [{"date": d.strftime(
                '%Y-%m-%d'), "price": round(float(r['Close']), 2)} for d, r in hist.iterrows()]

            resultado = {
                "success": True,
                "current_price": ultimo_precio,
                "change_pct": change_pct,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "history": history_data
            }
            set_cached(cache_key, resultado)
            return jsonify(resultado)

        return jsonify({'success': False, 'error': 'No hay datos de mercado hoy'}), 404

    except Exception as e:
        app.logger.error(f"Error obteniendo Brent: {str(e)}")
        if is_rate_limit_error(e):
            return jsonify({'success': False, 'error': 'rate_limit', 'message': 'Yahoo Finance rate limit. Intentá en unos minutos.'}), 429
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mercado/resumen', methods=['GET'])
def mercado_resumen():
    cache_key = 'mercado_resumen'
    cached = get_cached(cache_key)
    if cached:
        return jsonify(cached)

    tickers = ["YPF", "VIST", "PAM", "TS", "CEPU", "TGS"]
    nombres = {
        "YPF": "YPF SA", "VIST": "Vista Energy", "PAM": "Pampa Energía",
        "TS": "Tenaris", "CEPU": "Central Puerto", "TGS": "Transportadora Gas del Sur"
    }
    resultados = []
    try:
        for t in tickers:
            try:
                stock = yf.Ticker(t)
                hist = stock.history(period="5d")
                if len(hist) >= 2:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change = ((current - prev) / prev) * 100
                    resultados.append({
                        "ticker": t, "nombre": nombres.get(t, t),
                        "precio": round(current, 2), "variacion": round(change, 2)
                    })
            except Exception:
                pass  # saltar tickers individuales que fallen

        if not resultados:
            return jsonify({"success": False, "error": "rate_limit", "message": "Yahoo Finance rate limit. Intentá en unos minutos."}), 429

        resultado = {"success": True, "data": resultados}
        set_cached(cache_key, resultado)
        return jsonify(resultado)

    except Exception as e:
        app.logger.error(f"Error obteniendo resumen mercado: {str(e)}")
        if is_rate_limit_error(e):
            return jsonify({"success": False, "error": "rate_limit", "message": "Yahoo Finance rate limit. Intentá en unos minutos."}), 429
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/mercado/historico', methods=['GET'])
def mercado_historico():
    try:
        ticker = request.args.get('ticker', 'YPF')
        period = request.args.get('period', '1mo')
        cache_key = f'historico_{ticker}_{period}'

        cached = get_cached(cache_key)
        if cached:
            return jsonify(cached)

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return jsonify({'success': False, 'error': 'Sin datos'}), 404

        ultimo_precio = round(float(hist['Close'].iloc[-1]), 2)
        open_price = round(float(hist['Open'].iloc[-1]), 2)
        high_price = round(float(hist['High'].max()), 2)
        low_price = round(float(hist['Low'].min()), 2)
        primer_precio = float(hist['Close'].iloc[0])
        change_pct = round(((ultimo_precio - primer_precio) /
                           primer_precio) * 100, 2) if primer_precio > 0 else 0.0
        history_data = [{"date": d.strftime(
            '%Y-%m-%d'), "price": round(float(r['Close']), 2)} for d, r in hist.iterrows()]

        resultado = {
            "success": True,
            "current_price": ultimo_precio,
            "change_pct": change_pct,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "history": history_data
        }
        set_cached(cache_key, resultado)
        return jsonify(resultado)

    except Exception as e:
        app.logger.error(f"Error obteniendo histórico de {ticker}: {str(e)}")
        if is_rate_limit_error(e):
            return jsonify({'success': False, 'error': 'rate_limit', 'message': 'Yahoo Finance rate limit. Intentá en unos minutos.'}), 429
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Demo endpoints ────────────────────────────────────────────────

@app.route('/datos-demo/empresas', methods=['GET'])
def demo_empresas():
    """Lista de empresas disponibles en el parquet demo. Lee solo 2 columnas."""
    global _demo_empresas_cache
    if not os.path.exists(DEMO_PARQUET):
        return jsonify({'success': False, 'error': 'Demo no disponible'}), 503
    try:
        if _demo_empresas_cache is None:
            df = pd.read_parquet(DEMO_PARQUET, columns=[
                                 'idempresa', 'empresa'])
            df['idempresa'] = df['idempresa'].astype(
                str).str.strip().replace('nan', '')
            df['empresa'] = df['empresa'].astype(
                str).str.strip().replace('nan', '')
            _demo_empresas_cache = (
                df[df['idempresa'] != '']
                .drop_duplicates(subset='idempresa')
                .sort_values('empresa')[['idempresa', 'empresa']]
                .to_dict(orient='records')
            )
            app.logger.info(
                f"Demo empresas cacheadas: {len(_demo_empresas_cache)}")
        return jsonify({'success': True, 'empresas': _demo_empresas_cache})
    except Exception as e:
        app.logger.error(f"Error demo empresas: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/datos-demo/filtrar', methods=['GET'])
def demo_filtrar():
    """Filtra el parquet demo por idempresa usando predicate pushdown (sin cargar todo en RAM)."""
    if not os.path.exists(DEMO_PARQUET):
        return jsonify({'success': False, 'error': 'Demo no disponible'}), 503
    idempresa = request.args.get('idempresa', '').strip()
    if not idempresa:
        return jsonify({'success': False, 'error': 'idempresa requerido'}), 400
    try:
        df = pd.read_parquet(DEMO_PARQUET, filters=[
                             ('idempresa', '==', idempresa)])
        if df.empty:
            return jsonify({'success': False, 'error': f'No se encontraron datos para idempresa={idempresa}'}), 404
        df = optimizar_df(df)
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(str)
        empresa_nombre = df['empresa'].iloc[0] if 'empresa' in df.columns else idempresa
        historico, catalogo = procesar_df(df)
        return jsonify({
            'success':   True,
            'historico': historico,
            'catalogo':  catalogo,
            'empresa':   empresa_nombre,
            'total':     len(historico),
            'pozos':     len(catalogo),
        })
    except Exception as e:
        app.logger.error(f"Error filtrando demo: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Servidor de Análisis Petrolero NLP")
    print("=====================================")
    print(f"API Anthropic: {'Configurada' if API_KEY else 'NO configurada'}")
    print("URL: http://localhost:5000")
    print("=====================================")
    app.run(debug=True, host='0.0.0.0', port=5000)
