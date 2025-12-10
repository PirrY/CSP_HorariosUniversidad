import pandas as pd
import numpy as np
import json
import copy
import time
import os
from datetime import datetime
from collections import defaultdict


class HorarioCSP:
    def __init__(self, cursos, aulas, franjas_horarias):
        self.cursos = cursos
        self.aulas = aulas
        self.franjas = franjas_horarias
        self.dominios_iniciales = self._generar_dominios_iniciales()
        self.nodos_explorados = 0
        self.backtracks = 0

    def _generar_dominios_iniciales(self):
        dominios = {}
        for curso_id, curso_info in self.cursos.items():
            dominio = []
            duracion = curso_info['duracion']
            inscritos = curso_info['inscritos']
            for aula_id, aula_info in self.aulas.items():
                if aula_info['capacidad'] < inscritos:
                    continue
                for franja in self.franjas:
                    if self._duracion_franja(franja) >= duracion:
                        dominio.append((aula_id, franja))
            dominios[curso_id] = dominio
        return dominios

    def _duracion_franja(self, franja):
        dia, hora_inicio, hora_fin = franja
        fmt = '%H:%M'
        inicio = datetime.strptime(hora_inicio, fmt)
        fin = datetime.strptime(hora_fin, fmt)
        return (fin - inicio).total_seconds() / 3600

    def es_consistente(self, curso_id, valor, asignacion):
        aula, franja = valor
        docente = self.cursos[curso_id]['docente']
        for otro_curso_id, (otra_aula, otra_franja) in asignacion.items():
            if aula == otra_aula and self._franjas_solapan(franja, otra_franja):
                return False
            otro_docente = self.cursos[otro_curso_id]['docente']
            if docente == otro_docente and docente != '-':
                if self._franjas_solapan(franja, otra_franja):
                    return False
        return True

    def _franjas_solapan(self, franja1, franja2):
        dia1, inicio1, fin1 = franja1
        dia2, inicio2, fin2 = franja2
        if dia1 != dia2:
            return False
        fmt = '%H:%M'
        t1_inicio = datetime.strptime(inicio1, fmt)
        t1_fin = datetime.strptime(fin1, fmt)
        t2_inicio = datetime.strptime(inicio2, fmt)
        t2_fin = datetime.strptime(fin2, fmt)
        return not (t1_fin <= t2_inicio or t2_fin <= t1_inicio)

    def calcular_costo(self, asignacion):
        if not asignacion:
            return 0
        costo_total = 0
        costo_total += 10 * self._penalizacion_huecos_docentes(asignacion)
        costo_total += 5 * self._penalizacion_desbalance_dias(asignacion)
        costo_total += 3 * self._penalizacion_desbalance_aulas(asignacion)
        costo_total += 2 * self._penalizacion_sesgo_franjas(asignacion)
        return costo_total

    def _penalizacion_huecos_docentes(self, asignacion):
        huecos_por_docente = defaultdict(list)
        for curso_id, (aula, franja) in asignacion.items():
            docente = self.cursos[curso_id]['docente']
            if docente == '-':
                continue
            dia, inicio, fin = franja
            huecos_por_docente[(docente, dia)].append((inicio, fin))
        total_huecos = 0
        for (docente, dia), clases in huecos_por_docente.items():
            clases_ordenadas = sorted(clases, key=lambda x: x[0])
            for i in range(len(clases_ordenadas) - 1):
                fin_actual = datetime.strptime(clases_ordenadas[i][1], '%H:%M')
                inicio_siguiente = datetime.strptime(clases_ordenadas[i+1][0], '%H:%M')
                hueco = (inicio_siguiente - fin_actual).total_seconds() / 3600
                if hueco > 0:
                    total_huecos += hueco
        return total_huecos

    def _penalizacion_desbalance_dias(self, asignacion):
        clases_por_dia = defaultdict(int)
        for curso_id, (aula, franja) in asignacion.items():
            dia, _, _ = franja
            clases_por_dia[dia] += 1
        if not clases_por_dia:
            return 0
        valores = list(clases_por_dia.values())
        return np.std(valores)

    def _penalizacion_desbalance_aulas(self, asignacion):
        horas_por_aula = defaultdict(float)
        for curso_id, (aula, franja) in asignacion.items():
            duracion = self._duracion_franja(franja)
            horas_por_aula[aula] += duracion
        if not horas_por_aula:
            return 0
        valores = list(horas_por_aula.values())
        promedio = np.mean(valores)
        if promedio == 0:
            return 0
        desviacion = np.std(valores)
        return desviacion / promedio if promedio > 0 else 0

    def _penalizacion_sesgo_franjas(self, asignacion):
        clases_por_franja = {'MAÑANA': 0, 'TARDE': 0, 'NOCHE': 0}
        for curso_id, (aula, franja) in asignacion.items():
            dia, inicio, fin = franja
            hora = datetime.strptime(inicio, '%H:%M').hour
            if 7 <= hora < 12:
                clases_por_franja['MAÑANA'] += 1
            elif 12 <= hora < 17:
                clases_por_franja['TARDE'] += 1
            elif 17 <= hora < 21:
                clases_por_franja['NOCHE'] += 1
        valores = list(clases_por_franja.values())
        if not valores or max(valores) == 0:
            return 0
        return max(valores) - min(valores)


def backtracking_fc(csp, asignacion=None, dominios=None):
    if asignacion is None:
        asignacion = {}
    if dominios is None:
        dominios = copy.deepcopy(csp.dominios_iniciales)

    csp.nodos_explorados += 1

    if len(asignacion) == len(csp.cursos):
        return asignacion

    curso = seleccionar_variable_mrv(csp, asignacion, dominios)
    valores = list(dominios[curso])

    for valor in valores:
        if csp.es_consistente(curso, valor, asignacion):
            asignacion[curso] = valor
            dominios_guardados = copy.deepcopy(dominios)

            if forward_check(csp, curso, valor, asignacion, dominios):
                resultado = backtracking_fc(csp, asignacion, dominios)
                if resultado is not None:
                    return resultado

            del asignacion[curso]
            dominios = dominios_guardados
            csp.backtracks += 1

    return None


def seleccionar_variable_mrv(csp, asignacion, dominios):
    no_asignadas = [curso for curso in csp.cursos if curso not in asignacion]
    if not no_asignadas:
        return None
    return min(no_asignadas, key=lambda c: len(dominios[c]))


def forward_check(csp, curso_asignado, valor_asignado, asignacion, dominios):
    aula_asignada, franja_asignada = valor_asignado
    docente_asignado = csp.cursos[curso_asignado]['docente']

    for curso in csp.cursos:
        if curso in asignacion:
            continue

        docente_curso = csp.cursos[curso]['docente']
        dominio_filtrado = []

        for (aula, franja) in dominios[curso]:
            consistente = True
            if aula == aula_asignada and csp._franjas_solapan(franja, franja_asignada):
                consistente = False
            if docente_curso == docente_asignado and docente_curso != '-':
                if csp._franjas_solapan(franja, franja_asignada):
                    consistente = False
            if consistente:
                dominio_filtrado.append((aula, franja))

        dominios[curso] = dominio_filtrado
        if not dominios[curso]:
            return False
    return True


def cargar_datos_trabajo(ruta_csv):
    return pd.read_csv(ruta_csv)


def construir_instancia_csp(df):
    cursos = {}
    for idx, row in df.iterrows():
        curso_id = row['ID_Curso']
        if curso_id not in cursos:
            cursos[curso_id] = {
                'nombre': row['Asignatura'],
                'docente': row['Docente'] if pd.notna(row['Docente']) else '-',
                'duracion': row['Duracion_horas'],
                'inscritos': int(row['Alumnos Inscritos']) if pd.notna(row['Alumnos Inscritos']) else 0,
                'capacidad_necesaria': int(row['Alumnos Inscritos']) if pd.notna(row['Alumnos Inscritos']) else 0
            }

    aulas = {}
    for idx, row in df.iterrows():
        aula_id = row['Aula']
        if pd.notna(aula_id) and aula_id not in aulas:
            aulas[aula_id] = {
                'capacidad': int(row['Capacidad Salon']) if pd.notna(row['Capacidad Salon']) else 50,
                'edificio': row['Edificio'] if pd.notna(row['Edificio']) else 'Desconocido'
            }

    dias = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES']
    horarios_2h = [
        ('07:00', '09:00'), ('09:00', '11:00'), ('11:00', '13:00'),
        ('14:00', '16:00'), ('16:00', '18:00'), ('18:00', '20:00'),
    ]
    horarios_3h = [
        ('07:00', '10:00'), ('09:00', '12:00'), ('10:00', '13:00'),
        ('14:00', '17:00'), ('15:00', '18:00'), ('17:00', '20:00'),
    ]

    franjas = []
    for dia in dias:
        for inicio, fin in horarios_2h:
            franjas.append((dia, inicio, fin))
        for inicio, fin in horarios_3h:
            franjas.append((dia, inicio, fin))

    return HorarioCSP(cursos, aulas, franjas)


def resolver_csp(csp):
    inicio = time.time()
    solucion = backtracking_fc(csp)
    tiempo_total = time.time() - inicio
    return solucion, {
        'tiempo': tiempo_total,
        'nodos': csp.nodos_explorados,
        'backtracks': csp.backtracks,
        'exito': solucion is not None
    }


def guardar_solucion(csp, solucion, ruta_salida='../resultados/horario_optimizado.csv'):
    os.makedirs(os.path.dirname(ruta_salida) if os.path.dirname(ruta_salida) else '.', exist_ok=True)
    registros = []
    for curso_id, (aula, franja) in solucion.items():
        dia, inicio, fin = franja
        curso_info = csp.cursos[curso_id]
        registros.append({
            'ID_Curso': curso_id,
            'Asignatura': curso_info['nombre'],
            'Docente': curso_info['docente'],
            'Dia': dia,
            'Hora_Inicio': inicio,
            'Hora_Fin': fin,
            'Aula': aula,
            'Alumnos': curso_info['inscritos'],
            'Capacidad_Aula': csp.aulas[aula]['capacidad']
        })
    df_solucion = pd.DataFrame(registros)
    df_solucion.to_csv(ruta_salida, index=False)
    return df_solucion


if __name__ == '__main__':
    if os.path.exists('../resultados/datos_trabajo.csv'):
        ruta_datos = '../resultados/datos_trabajo.csv'
    elif os.path.exists('resultados/datos_trabajo.csv'):
        ruta_datos = 'resultados/datos_trabajo.csv'
    else:
        raise FileNotFoundError("No se encontro datos_trabajo.csv")

    df = cargar_datos_trabajo(ruta_datos)
    csp = construir_instancia_csp(df)
    solucion, stats = resolver_csp(csp)

    if solucion:
        guardar_solucion(csp, solucion)
        stats['costo_total'] = csp.calcular_costo(solucion)
        with open('../resultados/estadisticas_csp.json', 'w') as f:
            json.dump(stats, f, indent=2)
