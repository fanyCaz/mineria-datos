from random import choice
import csv

numericas = [20,30,40,50,60] # edad

booleana = [False, True]

nominal = ["San Nicolas", "Monterrey", "Guadalupe", "Apodaca", "Escobedo"]

ordinaria = ["Primaria", "Secundaria", "Preparatoria", "Superior"]

registros = []

salida = ["Aprobado", "Denegado"]

for i in range(100):
    edad = choice(numericas)
    municipio = choice(nominal)
    civil = choice(booleana)
    escuela = choice(ordinaria)
    status = choice(salida)
    registro = {"ID": i+1, "Edad" : edad, "Casad@" : civil, "Escolaridad" : escuela, "Municipio": municipio, "Credito": status}
    registros.append(registro)

with open('registrosCredito.csv', 'w', newline='') as csvfile:
    fieldnames = ["ID","Edad", "Casad@", "Escolaridad", "Municipio", "Credito"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for registro in registros:
        writer.writerow(registro)
    
    
        
