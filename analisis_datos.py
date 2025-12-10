import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def cargar_datos(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        content = f.read()
    from io import StringIO
    tables = pd.read_html(StringIO(content))
    return max(tables, key=len)


def limpiar_datos(df):
    df_clean = df.copy()
    dias_map = {
        'LUNES': 'LUNES',
        'MARTES': 'MARTES',
        'MIÉRCOLES': 'MIERCOLES',
        'MIÉRCOLES': 'MIERCOLES',
        'JUEVES': 'JUEVES',
        'VIERNES': 'VIERNES',
        'SÁBADO': 'SABADO',
        'SÁBADO': 'SABADO',
        'DOMINGO': 'DOMINGO'
    }
    df_clean['Dia Semana'] = df_clean['Dia Semana'].map(lambda x: dias_map.get(x, x) if pd.notna(x) else x)
    df_clean['Hora_Inicio_dt'] = pd.to_datetime(df_clean['Hora Inicio'], format='%H:%M', errors='coerce')
    df_clean['Hora_Fin_dt'] = pd.to_datetime(df_clean['Hora Fin'], format='%H:%M', errors='coerce')
    df_clean['Duracion_horas'] = (df_clean['Hora_Fin_dt'] - df_clean['Hora_Inicio_dt']).dt.total_seconds() / 3600
    df_clean = df_clean.dropna(subset=['Asignatura', 'Dia Semana', 'Hora Inicio'])
    df_clean['ID_Curso'] = df_clean['Codigo Asigantura'].astype(str) + '_' + df_clean['Grupo'].astype(str)
    return df_clean


def seleccionar_departamento(df, departamento=None):
    dept_counts = df['Departamento'].value_counts()
    if departamento is None:
        for dept, count in dept_counts.items():
            if 40 <= count <= 80:
                departamento = dept
                break
        if departamento is None:
            departamento = dept_counts.index[0]
    df_dept = df[df['Departamento'] == departamento].copy()
    return df_dept, departamento


def calcular_huecos_docentes(df):
    huecos_por_docente = {}
    horas_totales_hueco = 0

    for docente, grupo in df.groupby('Docente'):
        if docente == '-' or pd.isna(docente):
            continue

        huecos_docente = 0
        for dia, clases_dia in grupo.groupby('Dia Semana'):
            if pd.isna(dia):
                continue

            clases_ordenadas = clases_dia.sort_values('Hora_Inicio_dt')
            for i in range(len(clases_ordenadas) - 1):
                fin_clase_actual = clases_ordenadas.iloc[i]['Hora_Fin_dt']
                inicio_clase_siguiente = clases_ordenadas.iloc[i + 1]['Hora_Inicio_dt']
                hueco = (inicio_clase_siguiente - fin_clase_actual).total_seconds() / 3600
                if hueco > 0:
                    huecos_docente += hueco

        huecos_por_docente[docente] = huecos_docente
        horas_totales_hueco += huecos_docente

    huecos_values = list(huecos_por_docente.values())
    promedio_huecos = np.mean(huecos_values) if huecos_values else 0
    std_huecos = np.std(huecos_values) if huecos_values else 0

    return {
        'promedio_huecos': promedio_huecos,
        'std_huecos': std_huecos,
        'total_huecos': horas_totales_hueco,
        'huecos_por_docente': huecos_por_docente
    }


def calcular_balance_dias(df):
    clases_por_dia = df['Dia Semana'].value_counts().to_dict()
    valores = list(clases_por_dia.values())
    promedio = np.mean(valores)
    std = np.std(valores)
    coef_variacion = (std / promedio * 100) if promedio > 0 else 0
    return {
        'clases_por_dia': clases_por_dia,
        'promedio': promedio,
        'std': std,
        'coef_variacion': coef_variacion
    }


def calcular_utilizacion_aulas(df):
    HORAS_DISPONIBLES_DIA = 14
    DIAS_SEMANA = 5
    HORAS_TOTALES_SEMANA = HORAS_DISPONIBLES_DIA * DIAS_SEMANA

    utilizacion_por_aula = {}
    for aula, grupo in df.groupby('Aula'):
        if pd.isna(aula) or aula == '-':
            continue
        horas_usadas = grupo['Duracion_horas'].sum()
        utilizacion = (horas_usadas / HORAS_TOTALES_SEMANA) * 100
        utilizacion_por_aula[aula] = utilizacion

    valores = list(utilizacion_por_aula.values())
    promedio_utilizacion = np.mean(valores) if valores else 0
    std_utilizacion = np.std(valores) if valores else 0
    subutilizadas = {k: v for k, v in utilizacion_por_aula.items() if v < 30}
    sobreutilizadas = {k: v for k, v in utilizacion_por_aula.items() if v > 70}

    return {
        'promedio_utilizacion': promedio_utilizacion,
        'std_utilizacion': std_utilizacion,
        'utilizacion_por_aula': utilizacion_por_aula,
        'aulas_subutilizadas': subutilizadas,
        'aulas_sobreutilizadas': sobreutilizadas
    }


def calcular_balance_franjas(df):
    def clasificar_franja(hora_dt):
        if pd.isna(hora_dt):
            return None
        hora = hora_dt.hour
        if 7 <= hora < 12:
            return 'MAÑANA'
        elif 12 <= hora < 17:
            return 'TARDE'
        elif 17 <= hora < 21:
            return 'NOCHE'
        else:
            return 'OTRO'

    df['Franja'] = df['Hora_Inicio_dt'].apply(clasificar_franja)
    clases_por_franja = df['Franja'].value_counts().to_dict()
    total_clases = sum(clases_por_franja.values())
    porcentajes = {k: (v / total_clases * 100) for k, v in clases_por_franja.items()}
    valores = [clases_por_franja.get(f, 0) for f in ['MAÑANA', 'TARDE', 'NOCHE']]
    sesgo = max(valores) - min(valores)

    return {
        'clases_por_franja': clases_por_franja,
        'porcentajes': porcentajes,
        'sesgo': sesgo
    }


def calcular_metricas_desde_csv(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df['Hora_Inicio_dt'] = pd.to_datetime(df['Hora_Inicio'], format='%H:%M', errors='coerce')
    df['Hora_Fin_dt'] = pd.to_datetime(df['Hora_Fin'], format='%H:%M', errors='coerce')
    df['Duracion_horas'] = (df['Hora_Fin_dt'] - df['Hora_Inicio_dt']).dt.total_seconds() / 3600
    df = df.rename(columns={'Dia': 'Dia Semana'})

    return {
        'huecos_docentes': calcular_huecos_docentes(df),
        'balance_dias': calcular_balance_dias(df),
        'utilizacion_aulas': calcular_utilizacion_aulas(df),
        'balance_franjas': calcular_balance_franjas(df)
    }


def calcular_mejoras(metricas_actual, metricas_opt):
    mejoras = {}

    actual_huecos = metricas_actual['huecos_docentes']['promedio_huecos']
    opt_huecos = metricas_opt['huecos_docentes']['promedio_huecos']
    mejora_huecos = ((actual_huecos - opt_huecos) / actual_huecos * 100) if actual_huecos > 0 else 0
    mejoras['huecos_docentes'] = {
        'actual': actual_huecos,
        'optimizado': opt_huecos,
        'mejora_porcentaje': mejora_huecos,
        'mejor': 'optimizado' if opt_huecos < actual_huecos else 'actual'
    }

    actual_std = metricas_actual['balance_dias']['std']
    opt_std = metricas_opt['balance_dias']['std']
    mejora_dias = ((actual_std - opt_std) / actual_std * 100) if actual_std > 0 else 0
    mejoras['balance_dias'] = {
        'actual': actual_std,
        'optimizado': opt_std,
        'mejora_porcentaje': mejora_dias,
        'mejor': 'optimizado' if opt_std < actual_std else 'actual'
    }

    actual_util = metricas_actual['utilizacion_aulas']['std_utilizacion']
    opt_util = metricas_opt['utilizacion_aulas']['std_utilizacion']
    mejora_util = ((actual_util - opt_util) / actual_util * 100) if actual_util > 0 else 0
    mejoras['balance_aulas'] = {
        'actual': actual_util,
        'optimizado': opt_util,
        'mejora_porcentaje': mejora_util,
        'mejor': 'optimizado' if opt_util < actual_util else 'actual'
    }

    actual_sesgo = metricas_actual['balance_franjas']['sesgo']
    opt_sesgo = metricas_opt['balance_franjas']['sesgo']
    mejora_sesgo = ((actual_sesgo - opt_sesgo) / actual_sesgo * 100) if actual_sesgo > 0 else 0
    mejoras['balance_franjas'] = {
        'actual': actual_sesgo,
        'optimizado': opt_sesgo,
        'mejora_porcentaje': mejora_sesgo,
        'mejor': 'optimizado' if opt_sesgo < actual_sesgo else 'actual'
    }

    return mejoras


def generar_visualizaciones_comparativas(metricas_actual, metricas_opt, mejoras, output_dir='../resultados/figuras'):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparacion: Horario Actual vs Horario Optimizado', fontsize=16, fontweight='bold')

    metricas_nombres = ['Actual', 'Optimizado']
    colores = ['#e74c3c', '#27ae60']

    ax = axes[0, 0]
    valores = [mejoras['huecos_docentes']['actual'], mejoras['huecos_docentes']['optimizado']]
    bars = ax.bar(metricas_nombres, valores, color=colores, alpha=0.7)
    ax.set_ylabel('Horas/semana')
    ax.set_title('Huecos en Horarios de Docentes')
    ax.set_ylim(0, max(valores) * 1.2)
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}h',
                ha='center', va='bottom', fontweight='bold')

    ax = axes[0, 1]
    valores = [mejoras['balance_dias']['actual'], mejoras['balance_dias']['optimizado']]
    bars = ax.bar(metricas_nombres, valores, color=colores, alpha=0.7)
    ax.set_ylabel('Desviacion estandar')
    ax.set_title('Balance de Distribucion por Dias')
    ax.set_ylim(0, max(valores) * 1.2)
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 0]
    valores = [mejoras['balance_aulas']['actual'], mejoras['balance_aulas']['optimizado']]
    bars = ax.bar(metricas_nombres, valores, color=colores, alpha=0.7)
    ax.set_ylabel('Desviacion Estandar (%)')
    ax.set_title('Balance en Utilizacion de Aulas')
    ax.set_ylim(0, max(valores) * 1.3)
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 1]
    valores = [mejoras['balance_franjas']['actual'], mejoras['balance_franjas']['optimizado']]
    bars = ax.bar(metricas_nombres, valores, color=colores, alpha=0.7)
    ax.set_ylabel('Diferencia (clases)')
    ax.set_title('Sesgo de Franjas Horarias')
    ax.set_ylim(0, max(valores) * 1.2)
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(val)}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparacion_metricas.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    tabla_data = [
        ['Metrica', 'Actual', 'Optimizado', 'Mejora', 'Ganador'],
        ['Huecos Docentes',
         f"{mejoras['huecos_docentes']['actual']:.1f}h",
         f"{mejoras['huecos_docentes']['optimizado']:.1f}h",
         f"{mejoras['huecos_docentes']['mejora_porcentaje']:+.1f}%",
         'Opt' if mejoras['huecos_docentes']['mejor'] == 'optimizado' else 'Actual'],
        ['Balance Dias',
         f"{mejoras['balance_dias']['actual']:.2f}",
         f"{mejoras['balance_dias']['optimizado']:.2f}",
         f"{mejoras['balance_dias']['mejora_porcentaje']:+.1f}%",
         'Opt' if mejoras['balance_dias']['mejor'] == 'optimizado' else 'Actual'],
        ['Balance Aulas',
         f"{mejoras['balance_aulas']['actual']:.1f}%",
         f"{mejoras['balance_aulas']['optimizado']:.1f}%",
         f"{mejoras['balance_aulas']['mejora_porcentaje']:+.1f}%",
         'Opt' if mejoras['balance_aulas']['mejor'] == 'optimizado' else 'Actual'],
        ['Sesgo Franjas',
         f"{int(mejoras['balance_franjas']['actual'])}",
         f"{int(mejoras['balance_franjas']['optimizado'])}",
         f"{mejoras['balance_franjas']['mejora_porcentaje']:+.1f}%",
         'Opt' if mejoras['balance_franjas']['mejor'] == 'optimizado' else 'Actual'],
    ]

    table = ax.table(cellText=tabla_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, 5):
        for j in range(5):
            if j == 4 and 'Opt' in tabla_data[i][j]:
                table[(i, j)].set_facecolor('#d5f4e6')
                table[(i, j)].set_text_props(weight='bold', color='green')

    plt.title('Tabla Comparativa de Metricas', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/tabla_comparativa.png', dpi=300, bbox_inches='tight')
    plt.close()


def generar_visualizaciones(df, metricas, output_dir='../resultados/figuras'):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    dias_orden = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO']
    dias_data = [metricas['balance_dias']['clases_por_dia'].get(d, 0) for d in dias_orden]
    plt.bar(dias_orden, dias_data, color='steelblue', alpha=0.7)
    plt.axhline(y=metricas['balance_dias']['promedio'], color='red', linestyle='--',
                label=f'Promedio: {metricas["balance_dias"]["promedio"]:.1f}')
    plt.xlabel('Dia de la semana')
    plt.ylabel('Numero de clases')
    plt.title('Distribucion de Clases por Dia - Horario Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribucion_dias_actual.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    franjas = ['MAÑANA', 'TARDE', 'NOCHE']
    franjas_data = [metricas['balance_franjas']['clases_por_franja'].get(f, 0) for f in franjas]
    colors = ['#FFD700', '#FF8C00', '#4B0082']
    plt.bar(franjas, franjas_data, color=colors, alpha=0.7)
    plt.xlabel('Franja horaria')
    plt.ylabel('Numero de clases')
    plt.title('Distribucion de Clases por Franja Horaria - Horario Actual')
    for i, (franja, valor) in enumerate(zip(franjas, franjas_data)):
        pct = metricas['balance_franjas']['porcentajes'].get(franja, 0)
        plt.text(i, valor + 1, f'{pct:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribucion_franjas_actual.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    dias_orden = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES']
    horas = range(7, 21)
    ocupacion = np.zeros((len(horas), len(dias_orden)))
    for idx, row in df.iterrows():
        dia = row['Dia Semana']
        if dia not in dias_orden:
            continue
        dia_idx = dias_orden.index(dia)
        hora_inicio = row['Hora_Inicio_dt'].hour if pd.notna(row['Hora_Inicio_dt']) else None
        hora_fin = row['Hora_Fin_dt'].hour if pd.notna(row['Hora_Fin_dt']) else None
        if hora_inicio is not None and hora_fin is not None:
            for hora in range(hora_inicio, min(hora_fin + 1, 21)):
                if 7 <= hora < 21:
                    ocupacion[hora - 7, dia_idx] += 1
    sns.heatmap(ocupacion, xticklabels=dias_orden, yticklabels=[f'{h}:00' for h in horas],
                cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Clases simultaneas'})
    plt.title('Heatmap de Ocupacion - Horario Actual')
    plt.xlabel('Dia de la semana')
    plt.ylabel('Hora del dia')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_actual.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    utilizaciones = list(metricas['utilizacion_aulas']['utilizacion_por_aula'].values())
    plt.hist(utilizaciones, bins=20, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(x=metricas['utilizacion_aulas']['promedio_utilizacion'],
                color='red', linestyle='--', linewidth=2,
                label=f'Promedio: {metricas["utilizacion_aulas"]["promedio_utilizacion"]:.1f}%')
    plt.axvline(x=30, color='orange', linestyle=':', label='Umbral subutilizacion (30%)')
    plt.axvline(x=70, color='darkred', linestyle=':', label='Umbral sobreutilizacion (70%)')
    plt.xlabel('Utilizacion (%)')
    plt.ylabel('Numero de aulas')
    plt.title('Distribucion de Utilizacion de Aulas - Horario Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/utilizacion_aulas_actual.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    if os.path.exists('../AuditoriaAcademica.xls'):
        ruta_datos = '../AuditoriaAcademica.xls'
    elif os.path.exists('AuditoriaAcademica.xls'):
        ruta_datos = 'AuditoriaAcademica.xls'
    else:
        raise FileNotFoundError("No se encontro el archivo AuditoriaAcademica.xls")

    df = cargar_datos(ruta_datos)
    df_clean = limpiar_datos(df)
    df_dept, departamento = seleccionar_departamento(df_clean)

    metricas = {
        'huecos_docentes': calcular_huecos_docentes(df_dept),
        'balance_dias': calcular_balance_dias(df_dept),
        'utilizacion_aulas': calcular_utilizacion_aulas(df_dept),
        'balance_franjas': calcular_balance_franjas(df_dept)
    }

    os.makedirs('../resultados', exist_ok=True)
    with open('../resultados/metricas_actual.json', 'w') as f:
        json.dump(metricas, f, indent=2)
    df_dept.to_csv('../resultados/datos_trabajo.csv', index=False)
    generar_visualizaciones(df_dept, metricas)
