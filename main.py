#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import analisis_datos
import csp_horarios


def main():
    if os.path.exists('../AuditoriaAcademica.xls'):
        ruta_datos = '../AuditoriaAcademica.xls'
    elif os.path.exists('AuditoriaAcademica.xls'):
        ruta_datos = 'AuditoriaAcademica.xls'
    else:
        print("Error: No se encontro el archivo AuditoriaAcademica.xls")
        sys.exit(1)

    df = analisis_datos.cargar_datos(ruta_datos)
    df_clean = analisis_datos.limpiar_datos(df)
    df_dept, departamento = analisis_datos.seleccionar_departamento(df_clean)

    metricas_actual = {
        'huecos_docentes': analisis_datos.calcular_huecos_docentes(df_dept),
        'balance_dias': analisis_datos.calcular_balance_dias(df_dept),
        'utilizacion_aulas': analisis_datos.calcular_utilizacion_aulas(df_dept),
        'balance_franjas': analisis_datos.calcular_balance_franjas(df_dept)
    }

    os.makedirs('../resultados', exist_ok=True)
    with open('../resultados/metricas_actual.json', 'w') as f:
        json.dump(metricas_actual, f, indent=2)
    df_dept.to_csv('../resultados/datos_trabajo.csv', index=False)
    analisis_datos.generar_visualizaciones(df_dept, metricas_actual)

    df_trabajo = csp_horarios.cargar_datos_trabajo('../resultados/datos_trabajo.csv')
    csp = csp_horarios.construir_instancia_csp(df_trabajo)
    solucion, stats = csp_horarios.resolver_csp(csp)

    if not solucion:
        print("Error: No se encontro solucion factible")
        sys.exit(1)

    csp_horarios.guardar_solucion(csp, solucion)
    stats['costo_total'] = csp.calcular_costo(solucion)
    with open('../resultados/estadisticas_csp.json', 'w') as f:
        json.dump(stats, f, indent=2)

    metricas_optimizado = analisis_datos.calcular_metricas_desde_csv(
        '../resultados/horario_optimizado.csv'
    )
    mejoras = analisis_datos.calcular_mejoras(metricas_actual, metricas_optimizado)
    analisis_datos.generar_visualizaciones_comparativas(
        metricas_actual, metricas_optimizado, mejoras
    )

    resultados_finales = {
        'metricas_actual': metricas_actual,
        'metricas_optimizado': metricas_optimizado,
        'mejoras': mejoras,
        'estadisticas_csp': stats
    }
    with open('../resultados/resultados_comparacion.json', 'w') as f:
        json.dump(resultados_finales, f, indent=2)

    print("Proceso completado exitosamente")


if __name__ == '__main__':
    main()
