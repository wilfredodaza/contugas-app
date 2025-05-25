from flask import Flask, render_template, request, send_file
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

# Carga inicial de datos
df = pd.read_csv('data/consolidated_data_con_anomalias.csv')#, parse_dates=['Fecha']

@app.route('/')
def dashboard():
    # KPIs generales
    total_clientes = df['id_cliente'].nunique()
    total_anomalias = df['anomaly'].sum()
    consumo_total = df['Volumen'].sum()

    return render_template('dashboard.html',
                           total_clientes=total_clientes,
                           total_anomalias=total_anomalias,
                           consumo_total=consumo_total)


@app.route('/consulta', methods=['GET', 'POST'])
def consulta():
    # Filtrar datos según parámetros recibidos
    clientes = sorted(df['id_cliente'].unique())
    filtro_cliente = request.args.get('cliente')
    filtro_fecha_ini = request.args.get('fecha_ini')
    filtro_fecha_fin = request.args.get('fecha_fin')
    filtro_anomalia = request.args.get('anomaly')

     # Si no hay cliente en los filtros, selecciona el primero del listado
    if not filtro_cliente:
        filtro_cliente = clientes[0]

    df_filtrado = df.copy()
    if filtro_cliente:
        df_filtrado = df_filtrado[df_filtrado['id_cliente'] == filtro_cliente]
    if filtro_fecha_ini and filtro_fecha_fin:
        df_filtrado = df_filtrado[
            (df_filtrado['Fecha'] >= filtro_fecha_ini) & 
            (df_filtrado['Fecha'] <= filtro_fecha_fin)]
    if filtro_anomalia == '1':
        df_filtrado = df_filtrado[df_filtrado['anomaly'] == 1]

    # Ejemplo: gráfico de líneas con anomalías
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtrado['Fecha'], y=df_filtrado['Volumen'], 
                             mode='lines', name='Volumen'))
    fig.add_trace(go.Scatter(x=df_filtrado[df_filtrado['anomaly']==1]['Fecha'],
                             y=df_filtrado[df_filtrado['anomaly']==1]['Volumen'],
                             mode='markers', name='Anomalía', marker=dict(color='red')))
    graph_html = pio.to_html(fig, full_html=False)

    return render_template('consulta.html',
                           clientes=clientes,
                           df=df_filtrado,
                           graph_html=graph_html)
                           

@app.route('/cliente/<id_cliente>')
def cliente(id_cliente):
    df_cliente = df[df['id_cliente'] == id_cliente]
    # KPIs, gráficos, etc
    return render_template('cliente.html', df=df_cliente, cliente=id_cliente)

@app.route('/alertas')
def alertas():
    alertas_df = df[df['anomaly'] == 1]
    return render_template('alertas.html', alertas=alertas_df)
"""
# Para descarga CSV filtrado
@app.route('/descargar', methods=['POST'])
def descargar():
    filtro = request.form.get('filtro', '')
    # Generar CSV filtrado según filtro
    df_filtrado = df # ... aplicar filtros como arriba
    df_filtrado.to_csv('data/tmp_export.csv', index=False)
    return send_file('data/tmp_export.csv', as_attachment=True)
"""
@app.route('/evalua', methods=['GET', 'POST'])
def evalua():
    resultado = None
    detalles = {}
    clientes = sorted(df['id_cliente'].unique())
    if request.method == 'POST':
        try:
            presion = float(request.form['presion'])
            temperatura = float(request.form['temperatura'])
            volumen = float(request.form['volumen'])
            cliente = request.form['cliente']
            estado, detalles = report_reading_web(presion, temperatura, volumen, cliente)
            resultado = estado
        except Exception as e:
            resultado = f"Error: {str(e)}"
    return render_template('evalua.html', clientes=clientes, resultado=resultado, detalles=detalles)

# Parámetros globales
lags  = [1, 2, 3]
vars3 = ['Presion', 'Temperatura', 'Volumen']
lag_feats = [f'{v}_lag{lag}' for lag in lags for v in vars3]

def report_reading_web(presion, temperatura, volumen, client_id):
    df_client = df[df['id_cliente'] == client_id].copy()
    if df_client.shape[0] < max(lags) + 1:
        raise ValueError("No hay suficientes datos históricos para calcular lags")
    scaler = StandardScaler()
    df_client[vars3] = scaler.fit_transform(df_client[vars3])
    for lag in lags:
        for v in vars3:
            df_client[f'{v}_lag{lag}'] = df_client[v].shift(lag)
    df_client = df_client.dropna().reset_index(drop=True)
    X_tr = df_client[lag_feats]
    y_tr = df_client[vars3]
    model = MLPRegressor(hidden_layer_sizes=(len(lag_feats) // 2,), max_iter=200, random_state=42)
    model.fit(X_tr, y_tr)
    preds_tr = model.predict(X_tr)
    mse_tr = np.mean((y_tr.values - preds_tr) ** 2, axis=1)
    thresh = np.percentile(mse_tr, 99)
    new_scaled = scaler.transform([[presion, temperatura, volumen]])[0]
    last_vals = df_client[vars3].iloc[-max(lags):].values
    new_lags = []
    for lag in lags:
        new_lags.extend(last_vals[-lag])
    X_new = np.array(new_lags).reshape(1, -1)
    pred_new = model.predict(X_new)[0]
    mse_new = np.mean((new_scaled - pred_new) ** 2)
    estado = "Anomalía" if mse_new > thresh else "Normal"
    detalles = {
        'cliente': client_id,
        'lectura_escalada': np.round(new_scaled, 3).tolist(),
        'mse_prediccion': round(mse_new, 5),
        'umbral': round(thresh, 5)
    }
    return estado, detalles


if __name__ == '__main__':
    app.run(debug=True)
