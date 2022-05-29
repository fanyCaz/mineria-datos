def support(total,n_both):
  if n_both == 0:
    return 0
  return n_both/total

def confidence(n_both,n_if):
  if n_if == 0:
    return 0
  return n_both/n_if

def lift(total,n_if,n_then,n_both):
  denom = (n_if/total) * (n_then/total) 
  if denom == 0:
    return 0
  return (n_both/total) / denom
