
file = "nih_chest_xrays2000.csv"

with open(file, 'r') as f:
  with open(f'new{file}', 'w') as nf:
    while True:
      line = f.readline()
      if not line:
        break
      elts = line.split(',')
      # print(elts)
      if len(elts) < 2:
        break
      if len(elts) > 2:
        additional_labels = elts[2:]
        for a in additional_labels:
          # print(f'a is {a}')
          nline = f'{elts[0]},{a.strip()}'
          # print(f'new line: {nline}')
          nf.write(f'{nline}\n')
      nf.write(f'{elts[0]},{elts[1].strip()}\n')
