"""
Optional: extra_constraint can receive an array of two elements,
the minimum value and the maximum value like: [0,2]
"""
def input_normalized(question: str, extra_constraint=None):
  res = input(question)
  try:
    if float(res) < 0:
      print('Ingresa un número positivo')
      return input_normalized(question)
    else:
      if extra_constraint:
        if float(res) < extra_constraint[0] or float(res) > extra_constraint[1]:
          print(f"Ingresa un número entre {extra_constraint[0]} y {extra_constraint[1]}")
          return input_normalized(question, extra_constraint)
      return float(res)
  except:
    print('Ingresa un número , porfavor')
    return input_normalized(question)
  return res

def imprimir_matriz(nombre, matriz):
  with open(f"{nombre}.txt", "w+") as archi:
    archi.writelines(f"{nombre}\n")
    archi.writelines("\n")
    archi.writelines("Matriz:\n")
    archi.writelines(f"{matriz}")

