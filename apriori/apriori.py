from metric_rules import support, confidence, lift

def apriori(metrics, minsup, data):
  metrics = list(filter(None,metrics)) # clean from empty dictionaries
  frequent_items = list(filter(lambda rule: rule['s'] >= minsup,metrics))
  print("Elementos frecuentes")
  print( [ f['rule'] for f in frequent_items ])
  I = frequent_items
  for k, items in enumerate(frequent_items):
    if k == len(frequent_items)-1:
      break
    else:
      # set of new rules aka Ck, combine two elements excluding duplicates
      new_rule_set = list( set( items['rule'].split(":") + frequent_items[k+1]['rule'].split(":") ))
      data_query = " & ".join(new_rule_set)
      result = data.query(data_query)
      sup = support(len(data), len(result) )
      if sup >= minsup:
        I.append(new_rule_set)
  print( "reglas finales")
  final_rules = list( set( [ i["rule"] for i in I]  ))
  print(final_rules)
  return I

